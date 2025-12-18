#!/usr/bin/env python3
"""
Synthetic AML transaction generator.
Generates transactions, accounts, and aggregate edges with optional suspicious patterns.
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from faker import Faker


fake = Faker()


@dataclass
class GenerationConfig:
    n_accounts: int = 10000
    n_transactions: int = 100000
    suspicious_ratio: Tuple[float, float] = (0.01, 0.03)
    structuring_share: float = 0.45
    circular_share: float = 0.25
    layering_share: float = 0.30
    start_date: dt.datetime = dt.datetime(2024, 1, 1)
    num_days: int = 120
    seed: int = 42


ACCOUNT_TYPES = ["individual", "company", "shell"]
ACCOUNT_TYPE_PROBS = [0.75, 0.20, 0.05]
CHANNELS = ["online_banking", "branch_cash", "atm", "mobile_wallet", "crypto_exchange"]
CURRENCIES = ["USD", "EUR", "GBP", "INR", "SGD"]
TX_TYPES = ["salary", "bill_pay", "retail", "merchant", "p2p"]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic AML transaction corpus")
    parser.add_argument("--n-accounts", type=int, default=10000, help="Number of accounts to create")
    parser.add_argument("--n-transactions", type=int, default=100000, help="Total transactions to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Directory for CSV outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Start date for timestamps (YYYY-MM-DD)")
    parser.add_argument("--num-days", type=int, default=120, help="Simulation horizon in days")
    parser.add_argument("--suspicious-low", type=float, default=0.01, help="Lower bound for suspicious ratio")
    parser.add_argument("--suspicious-high", type=float, default=0.03, help="Upper bound for suspicious ratio")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GenerationConfig:
    start_date = dt.datetime.fromisoformat(args.start_date)
    return GenerationConfig(
        n_accounts=args.n_accounts,
        n_transactions=args.n_transactions,
        suspicious_ratio=(args.suspicious_low, args.suspicious_high),
        start_date=start_date,
        num_days=args.num_days,
        seed=args.seed,
    )


def make_accounts(cfg: GenerationConfig) -> pd.DataFrame:
    account_ids = [f"ACC{i:06d}" for i in range(cfg.n_accounts)]
    types = np.random.choice(ACCOUNT_TYPES, size=cfg.n_accounts, p=ACCOUNT_TYPE_PROBS)
    created_offsets = np.random.randint(0, cfg.num_days, size=cfg.n_accounts)
    created_at = [cfg.start_date + dt.timedelta(days=int(d)) for d in created_offsets]
    balances = np.random.lognormal(mean=10, sigma=1.0, size=cfg.n_accounts)
    data = {
        "account_id": account_ids,
        "account_type": types,
        "created_at": created_at,
        "initial_balance": balances,
    }
    return pd.DataFrame(data)


def sample_country() -> str:
    # Use Faker countries for realism
    return fake.country()


def sample_currency(country: str) -> str:
    if country in {"United States", "Canada"}:
        return "USD"
    if country in {"Germany", "France", "Spain", "Italy"}:
        return "EUR"
    if country in {"United Kingdom", "Ireland"}:
        return "GBP"
    if country in {"India"}:
        return "INR"
    if country in {"Singapore"}:
        return "SGD"
    return random.choice(CURRENCIES)


def sample_timestamp(start: dt.datetime, num_days: int, window_hours: float | None = None) -> dt.datetime:
    base = start + dt.timedelta(days=random.uniform(0, num_days))
    if window_hours:
        base += dt.timedelta(hours=random.uniform(0, window_hours))
    hour_choice = np.random.choice([9, 11, 13, 15, 18, 20, 22], p=[0.10, 0.18, 0.15, 0.15, 0.20, 0.12, 0.10])
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return base.replace(hour=hour_choice, minute=minute, second=second)


def sample_amount(mean: float = 3.5, sigma: float = 0.8, multiplier: float = 1.0) -> float:
    return float(np.random.lognormal(mean=mean, sigma=sigma) * multiplier)


def ensure_distinct_pair(accounts: List[str]) -> Tuple[str, str]:
    sender = random.choice(accounts)
    receiver = random.choice(accounts)
    while receiver == sender:
        receiver = random.choice(accounts)
    return sender, receiver


def generate_normal_transactions(cfg: GenerationConfig, accounts: List[str], count: int, start_id: int) -> Tuple[List[Dict], int]:
    records: List[Dict] = []
    tx_id = start_id
    for _ in range(count):
        sender, receiver = ensure_distinct_pair(accounts)
        country = sample_country()
        record = {
            "tx_id": tx_id,
            "timestamp": sample_timestamp(cfg.start_date, cfg.num_days),
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": sample_amount(),
            "currency": sample_currency(country),
            "channel": random.choice(CHANNELS),
            "country": country,
            "tx_type": random.choice(TX_TYPES),
            "label": "normal",
            "scenario_id": "",
            "evidence": "",
        }
        records.append(record)
        tx_id += 1
    return records, tx_id


def inject_structuring(cfg: GenerationConfig, accounts: List[str], target_count: int, start_id: int, scenario_idx: int) -> Tuple[List[Dict], int, int]:
    records: List[Dict] = []
    tx_id = start_id
    remaining = target_count
    while remaining > 0:
        scenario_idx += 1
        target = random.choice(accounts)
        deposit_count = min(remaining, random.randint(50, 200))
        window_hours = random.uniform(24, 72)
        base_time = cfg.start_date + dt.timedelta(days=random.uniform(0, cfg.num_days))
        total_amount = float(np.random.uniform(20000, 120000))
        amounts = np.random.lognormal(mean=3.0, sigma=0.4, size=deposit_count)
        amounts = amounts / amounts.sum() * total_amount
        evidence = f"structuring into {target} with {deposit_count} deposits over {window_hours:.1f}h totaling {total_amount:.2f}"
        scenario_id = f"STRUCT-{scenario_idx:04d}"
        for j in range(deposit_count):
            sender, _ = ensure_distinct_pair(accounts + [target])
            records.append(
                {
                    "tx_id": tx_id,
                    "timestamp": base_time + dt.timedelta(hours=random.uniform(0, window_hours)),
                    "sender_id": sender,
                    "receiver_id": target,
                    "amount": float(amounts[j]),
                    "currency": "USD",
                    "channel": random.choice(["branch_cash", "atm", "online_banking"]),
                    "country": sample_country(),
                    "tx_type": "cash_deposit",
                    "label": "structuring",
                    "scenario_id": scenario_id,
                    "evidence": evidence,
                }
            )
            tx_id += 1
            remaining -= 1
            if remaining <= 0:
                break
    return records, tx_id, scenario_idx


def inject_circular(cfg: GenerationConfig, accounts: List[str], target_count: int, start_id: int, scenario_idx: int) -> Tuple[List[Dict], int, int]:
    records: List[Dict] = []
    tx_id = start_id
    remaining = target_count
    while remaining > 0:
        scenario_idx += 1
        cycle_len = random.randint(3, 6)
        nodes = random.sample(accounts, cycle_len)
        cycle_tx = min(remaining, cycle_len)
        base_time = cfg.start_date + dt.timedelta(days=random.uniform(0, cfg.num_days))
        window_hours = random.uniform(2, 24)
        scenario_id = f"CIRC-{scenario_idx:04d}"
        evidence = "->".join(nodes)
        for i in range(cycle_len):
            if cycle_tx <= 0:
                break
            sender = nodes[i]
            receiver = nodes[(i + 1) % cycle_len]
            records.append(
                {
                    "tx_id": tx_id,
                    "timestamp": base_time + dt.timedelta(hours=random.uniform(0, window_hours)),
                    "sender_id": sender,
                    "receiver_id": receiver,
                    "amount": float(np.random.uniform(5000, 40000)),
                    "currency": random.choice(["USD", "EUR", "GBP"]),
                    "channel": random.choice(CHANNELS),
                    "country": sample_country(),
                    "tx_type": "p2p",
                    "label": "circular",
                    "scenario_id": scenario_id,
                    "evidence": evidence,
                }
            )
            tx_id += 1
            remaining -= 1
            cycle_tx -= 1
    return records, tx_id, scenario_idx


def inject_layering(cfg: GenerationConfig, accounts: List[str], target_count: int, start_id: int, scenario_idx: int) -> Tuple[List[Dict], int, int]:
    records: List[Dict] = []
    tx_id = start_id
    remaining = target_count
    while remaining > 0:
        scenario_idx += 1
        path_len = random.randint(10, 16)
        if path_len > len(accounts):
            path_len = len(accounts)
        nodes = random.sample(accounts, path_len)
        hops = min(remaining, path_len - 1)
        base_time = cfg.start_date + dt.timedelta(days=random.uniform(0, cfg.num_days))
        window_hours = random.uniform(6, 36)
        scenario_id = f"LAYER-{scenario_idx:04d}"
        evidence = "->".join(nodes[: hops + 1])
        amount = float(np.random.uniform(10000, 80000))
        decay = np.linspace(1.0, 0.6, hops)
        for i in range(hops):
            records.append(
                {
                    "tx_id": tx_id,
                    "timestamp": base_time + dt.timedelta(hours=random.uniform(0, window_hours)),
                    "sender_id": nodes[i],
                    "receiver_id": nodes[i + 1],
                    "amount": amount * decay[i],
                    "currency": random.choice(CURRENCIES),
                    "channel": random.choice(CHANNELS),
                    "country": sample_country(),
                    "tx_type": "wire",
                    "label": "layering",
                    "scenario_id": scenario_id,
                    "evidence": evidence,
                }
            )
            tx_id += 1
            remaining -= 1
            if remaining <= 0:
                break
    return records, tx_id, scenario_idx


def split_sets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    def inject_if_empty(target: pd.DataFrame, source: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if (target["label"] != "normal").sum() == 0:
            movers = source[source["label"] != "normal"].head(10)
            if not movers.empty:
                target = pd.concat([target, movers], ignore_index=True)
                source = source.drop(movers.index)
        return target, source

    val, train = inject_if_empty(val, train)
    test, train = inject_if_empty(test, train)
    return train.sort_values("timestamp"), val.sort_values("timestamp"), test.sort_values("timestamp")


def aggregate_edges(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["sender_id", "receiver_id"])
        .agg(tx_count=("tx_id", "count"), total_amount=("amount", "sum"))
        .reset_index()
    )


def generate(cfg: GenerationConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    set_seeds(cfg.seed)
    accounts_df = make_accounts(cfg)
    account_ids = accounts_df["account_id"].tolist()

    suspicious_ratio = random.uniform(cfg.suspicious_ratio[0], cfg.suspicious_ratio[1])
    target_suspicious = max(1, int(cfg.n_transactions * suspicious_ratio))
    structuring_target = int(target_suspicious * cfg.structuring_share)
    circular_target = int(target_suspicious * cfg.circular_share)
    layering_target = target_suspicious - structuring_target - circular_target

    tx_records: List[Dict] = []
    next_tx_id = 1
    struct_records, next_tx_id, s_idx = inject_structuring(cfg, account_ids, structuring_target, next_tx_id, 0)
    tx_records.extend(struct_records)
    circ_records, next_tx_id, c_idx = inject_circular(cfg, account_ids, circular_target, next_tx_id, 0)
    tx_records.extend(circ_records)
    layer_records, next_tx_id, l_idx = inject_layering(cfg, account_ids, layering_target, next_tx_id, 0)
    tx_records.extend(layer_records)

    remaining_normals = max(0, cfg.n_transactions - len(tx_records))
    normal_records, next_tx_id = generate_normal_transactions(cfg, account_ids, remaining_normals, next_tx_id)
    tx_records.extend(normal_records)

    df = pd.DataFrame(tx_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    edges_df = aggregate_edges(df)
    stats = {
        "n_transactions": len(df),
        "n_accounts": len(accounts_df),
        "suspicious_ratio": (df["label"] != "normal").mean(),
        "structuring": (df["label"] == "structuring").sum(),
        "circular": (df["label"] == "circular").sum(),
        "layering": (df["label"] == "layering").sum(),
    }
    return df, accounts_df, edges_df, stats


def save_outputs(output_dir: Path, tx_df: pd.DataFrame, accounts_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tx_path = output_dir / "transaction.csv"
    accounts_path = output_dir / "accounts.csv"
    edges_path = output_dir / "edges.csv"
    tx_df.to_csv(tx_path, index=False)
    accounts_df.to_csv(accounts_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    train, val, test = split_sets(tx_df)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    tx_df, accounts_df, edges_df, stats = generate(cfg)
    save_outputs(args.output_dir, tx_df, accounts_df, edges_df)
    print("Generated dataset")
    print(f"Transactions: {stats['n_transactions']}")
    print(f"Accounts: {stats['n_accounts']}")
    print(f"Suspicious ratio: {stats['suspicious_ratio']:.4f}")
    print(f"Structuring: {stats['structuring']}, Circular: {stats['circular']}, Layering: {stats['layering']}")


if __name__ == "__main__":
    main()
