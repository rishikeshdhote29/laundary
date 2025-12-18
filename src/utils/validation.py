from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import (
    RAW_DIR,
    TRANSACTIONS_CSV,
    ACCOUNTS_CSV,
    EDGES_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    ALLOWED_LABELS,
    ALLOWED_CHANNELS,
    ALLOWED_CURRENCIES,
    TRAIN_FRACTION,
    VAL_FRACTION,
    TEST_FRACTION,
    FRACTION_TOLERANCE,
)


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    checks: Dict[str, bool]
    details: Dict[str, str]


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _check_required_columns(df: pd.DataFrame, cols: List[str]) -> Tuple[bool, str]:
    missing = [c for c in cols if c not in df.columns]
    return (len(missing) == 0, f"missing: {missing}" if missing else "")


def _class_ratio_ok(df: pd.DataFrame, low: float = 0.01, high: float = 0.03) -> Tuple[bool, str]:
    ratio = float((df["label"] != "normal").mean()) if len(df) else 0.0
    ok = low - 1e-6 <= ratio <= high + 1e-6
    return ok, f"ratio={ratio:.4f} expected [{low},{high}]"


def _split_fraction_ok(n_total: int, n_train: int, n_val: int, n_test: int) -> Tuple[bool, str]:
    if n_total == 0:
        return False, "empty dataset"
    fr_train = n_train / n_total
    fr_val = n_val / n_total
    fr_test = n_test / n_total
    ok = (
        abs(fr_train - TRAIN_FRACTION) <= FRACTION_TOLERANCE
        and abs(fr_val - VAL_FRACTION) <= FRACTION_TOLERANCE
        and abs(fr_test - TEST_FRACTION) <= FRACTION_TOLERANCE
    )
    msg = f"train={fr_train:.3f}, val={fr_val:.3f}, test={fr_test:.3f}"
    return ok, msg


def run_sanity_checks(raw_dir: Path = RAW_DIR) -> ValidationReport:
    checks: Dict[str, bool] = {}
    details: Dict[str, str] = {}

    # Existence
    for name, path in {
        "transactions_exists": raw_dir / "transaction.csv",
        "accounts_exists": raw_dir / "accounts.csv",
        "edges_exists": raw_dir / "edges.csv",
        "train_exists": raw_dir / "train.csv",
        "val_exists": raw_dir / "val.csv",
        "test_exists": raw_dir / "test.csv",
    }.items():
        ok = _exists(path)
        checks[name] = ok
        details[name] = str(path)

    if not all([checks[c] for c in [
        "transactions_exists",
        "accounts_exists",
        "edges_exists",
        "train_exists",
        "val_exists",
        "test_exists",
    ]]):
        return ValidationReport(False, checks, details)

    tx = pd.read_csv(raw_dir / "transaction.csv")
    accounts = pd.read_csv(raw_dir / "accounts.csv")
    edges = pd.read_csv(raw_dir / "edges.csv")
    train = pd.read_csv(raw_dir / "train.csv")
    val = pd.read_csv(raw_dir / "val.csv")
    test = pd.read_csv(raw_dir / "test.csv")

    # Required schemas
    tx_required = [
        "tx_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount",
        "currency",
        "channel",
        "country",
        "tx_type",
        "label",
        "scenario_id",
        "evidence",
    ]
    ok, msg = _check_required_columns(tx, tx_required)
    checks["tx_columns"] = ok
    details["tx_columns"] = msg

    acc_required = ["account_id", "account_type", "created_at", "initial_balance"]
    ok, msg = _check_required_columns(accounts, acc_required)
    checks["accounts_columns"] = ok
    details["accounts_columns"] = msg

    edges_required = ["sender_id", "receiver_id", "tx_count", "total_amount"]
    ok, msg = _check_required_columns(edges, edges_required)
    checks["edges_columns"] = ok
    details["edges_columns"] = msg

    # Basic content checks
    checks["tx_id_unique"] = tx["tx_id"].is_unique
    details["tx_id_unique"] = f"n={len(tx)}"

    checks["amount_positive"] = (tx["amount"] > 0).all()
    details["amount_positive"] = "all > 0"

    checks["sender_receiver_distinct"] = (tx["sender_id"] != tx["receiver_id"]).all()
    details["sender_receiver_distinct"] = "no self-transfers"

    checks["labels_allowed"] = set(tx["label"]).issubset(ALLOWED_LABELS)
    details["labels_allowed"] = f"labels={sorted(set(tx['label']))}"

    checks["channels_allowed"] = set(tx["channel"]).issubset(set(ALLOWED_CHANNELS))
    details["channels_allowed"] = f"channels={sorted(set(tx['channel']))}"

    checks["currencies_allowed"] = set(tx["currency"]).issubset(set(ALLOWED_CURRENCIES))
    details["currencies_allowed"] = f"currencies={sorted(set(tx['currency']))}"

    # Account references
    acc_ids = set(accounts["account_id"]) if len(accounts) else set()
    checks["sender_in_accounts"] = set(tx["sender_id"]).issubset(acc_ids)
    checks["receiver_in_accounts"] = set(tx["receiver_id"]).issubset(acc_ids)
    details["sender_in_accounts"] = f"unique={tx['sender_id'].nunique()}"
    details["receiver_in_accounts"] = f"unique={tx['receiver_id'].nunique()}"

    # Edge aggregation consistency
    agg = (
        tx.groupby(["sender_id", "receiver_id"]).agg(tx_count=("tx_id", "count"), total_amount=("amount", "sum")).reset_index()
    )
    merged = edges.merge(agg, on=["sender_id", "receiver_id"], suffixes=("_gen", "_agg"))
    checks["edges_match"] = (
        len(merged) == len(edges)
        and (merged["tx_count_gen"] == merged["tx_count_agg"]).all()
        and (merged["total_amount_gen"] - merged["total_amount_agg"]).abs().max() < 1e-6
    )
    details["edges_match"] = f"rows={len(edges)}"

    # Splits: proportions and suspicious coverage
    n_total = len(tx)
    ok_frac, msg_frac = _split_fraction_ok(n_total, len(train), len(val), len(test))
    checks["split_fractions_ok"] = ok_frac
    details["split_fractions_ok"] = msg_frac

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        key = f"{split_name}_has_suspicious"
        checks[key] = (split_df["label"] != "normal").sum() > 0
        details[key] = f"count={(split_df['label'] != 'normal').sum()}"

    ok_ratio, msg_ratio = _class_ratio_ok(tx)
    checks["suspicious_ratio"] = ok_ratio
    details["suspicious_ratio"] = msg_ratio

    passed = all(checks.values())
    return ValidationReport(passed=passed, checks=checks, details=details)
