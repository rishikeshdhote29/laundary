#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import RAW_DIR, REPORTS_DIR, FUSION_WEIGHTS, ensure_dirs


PATTERNS = ["structuring", "circular", "layering"]


def load_fused() -> pd.DataFrame:
    fused_path = REPORTS_DIR / "fused_scores.csv"
    if not fused_path.exists():
        raise FileNotFoundError(f"Missing fused scores at {fused_path}. Run scripts/fuse_scores.py first.")
    df = pd.read_csv(fused_path)
    if "fusion_score" not in df.columns:
        raise ValueError("fused_scores.csv missing fusion_score column")
    return df


def load_ground_truth(split: str = "val") -> pd.DataFrame:
    split_path = RAW_DIR / f"{split}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split}.csv in {RAW_DIR}")
    df = pd.read_csv(split_path, parse_dates=["timestamp"])
    parts = []
    for col in ["sender_id", "receiver_id"]:
        sub = df[[col, "label"]].rename(columns={col: "account_id"})
        parts.append(sub)
    acc = pd.concat(parts, ignore_index=True)
    out = acc.pivot_table(index="account_id", columns="label", aggfunc=len, fill_value=0)
    out = out.reset_index()
    for p in PATTERNS:
        if p not in out.columns:
            out[p] = 0
    out["suspicious"] = out[[p for p in PATTERNS]].sum(axis=1) > 0
    return out[["account_id"] + PATTERNS + ["suspicious"]]


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def best_f1_threshold(df: pd.DataFrame, score_col: str, label_col: str) -> Tuple[float, Dict[str, float]]:
    # Sweep unique scores as thresholds (descending)
    uniq = sorted(df[score_col].unique(), reverse=True)
    best = (0.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
    y_true = df[label_col].astype(bool).to_numpy()
    for t in uniq:
        y_pred = (df[score_col] >= t).to_numpy()
        tp = int((y_pred & y_true).sum())
        fp = int((y_pred & ~y_true).sum())
        fn = int((~y_pred & y_true).sum())
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        if f1 > best[0]:
            best = (f1, {"threshold": float(t), "precision": prec, "recall": rec, "f1": f1})
    return best[1].get("threshold", 0.0), best[1]


def precision_at_k(ranked_df: pd.DataFrame, label_set: set, k: int) -> float:
    topk = ranked_df.head(k)
    if topk.empty:
        return 0.0
    hits = topk["account_id"].isin(label_set).sum()
    return hits / len(topk)


def build_evidence(tx: pd.DataFrame, cycles: List[str], accounts: List[str]) -> Dict[str, str]:
    evidence: Dict[str, str] = {}
    tx_susp = tx[tx["label"] != "normal"].copy()
    tx_susp.sort_values("timestamp", inplace=True)
    # Precompute cycle lines per account for circular evidence
    cycle_map: Dict[str, str] = {}
    for line in cycles:
        parts = line.strip().split("->")
        for p in parts:
            cycle_map.setdefault(p, line.strip())

    for acc in accounts:
        acc_tx = tx_susp[(tx_susp["sender_id"] == acc) | (tx_susp["receiver_id"] == acc)]
        if acc_tx.empty:
            evidence[acc] = "No labeled suspicious transactions found; flagged by model."
            continue
        pieces = []
        for pattern in PATTERNS:
            sub = acc_tx[acc_tx["label"] == pattern]
            if sub.empty:
                continue
            count = len(sub)
            total = sub["amount"].sum()
            span_hours = (sub["timestamp"].max() - sub["timestamp"].min()).total_seconds() / 3600 if count > 1 else 0
            if pattern == "structuring":
                pieces.append(f"{count} deposits totaling {total:,.0f} over {span_hours:.1f}h")
            elif pattern == "circular":
                cyc_line = cycle_map.get(acc, None)
                if cyc_line:
                    pieces.append(f"cycle detected: {cyc_line}")
                else:
                    pieces.append(f"{count} circular legs totaling {total:,.0f}")
            elif pattern == "layering":
                pieces.append(f"{count} layering hops totaling {total:,.0f} over {span_hours:.1f}h")
        evidence[acc] = "; ".join(pieces) if pieces else "Flagged by model; no labeled evidence in split."
    return evidence


def load_cycle_lines() -> List[str]:
    path = REPORTS_DIR / "circular_cycles_evidence.txt"
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def feature_importance(fused: pd.DataFrame, val_labels: pd.DataFrame) -> pd.DataFrame:
    # Try logistic regression; fallback to fusion weights
    if not val_labels.empty and val_labels["suspicious"].sum() > 0:
        try:
            from sklearn.linear_model import LogisticRegression
            Xy = fused.merge(val_labels[["account_id", "suspicious"]], on="account_id", how="inner")
            if not Xy.empty and Xy["suspicious"].sum() > 0:
                X = Xy[["structuring_score", "circular_score", "layering_score"]].values
                y = Xy["suspicious"].astype(int).values
                model = LogisticRegression(max_iter=200)
                model.fit(X, y)
                coefs = model.coef_[0]
                return pd.DataFrame({
                    "feature": ["structuring_score", "circular_score", "layering_score"],
                    "importance": coefs,
                })
        except Exception:
            pass
    # fallback: weights
    return pd.DataFrame({
        "feature": ["structuring_score", "circular_score", "layering_score"],
        "importance": [FUSION_WEIGHTS.get("structuring", 0.0), FUSION_WEIGHTS.get("circular", 0.0), FUSION_WEIGHTS.get("layering", 0.0)],
    })


def main() -> None:
    ensure_dirs()
    fused = load_fused()
    val_labels = load_ground_truth("val")

    # Merge labels
    data = fused.merge(val_labels, on="account_id", how="left").fillna({"structuring": 0, "circular": 0, "layering": 0, "suspicious": 0})
    data["suspicious"] = data["suspicious"].astype(bool)
    for p in PATTERNS:
        data[p] = data[p].astype(bool)

    metrics: Dict[str, Dict[str, float]] = {}

    # Overall best-F1 threshold on fusion_score vs suspicious
    thr, overall = best_f1_threshold(data, "fusion_score", "suspicious")
    metrics["overall_best_f1"] = overall

    # Pattern-specific best-F1 using pattern_tag as prediction and pattern ground truth
    for p in PATTERNS:
        subset = data.copy()
        subset[f"pred_{p}"] = subset["pattern_tag"] == p
        y_true = subset[p].to_numpy()
        y_pred = subset[f"pred_{p}"].to_numpy()
        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        metrics[f"{p}_tag_metrics"] = {"precision": prec, "recall": rec, "f1": f1}

    # Precision@k for suspicious
    label_set = set(data.loc[data["suspicious"], "account_id"])
    p_at: Dict[str, float] = {}
    for k in [50, 100, 200]:
        p_at[f"precision@{k}"] = precision_at_k(data.sort_values("fusion_score", ascending=False), label_set, k)
    metrics["precision_at_k"] = p_at

    # Evidence for top-K fused
    cycles = load_cycle_lines()
    top_accounts = data.sort_values("fusion_score", ascending=False).head(50)["account_id"].tolist()
    tx_val = pd.read_csv(RAW_DIR / "val.csv", parse_dates=["timestamp"]) if (RAW_DIR / "val.csv").exists() else pd.DataFrame()
    evidence = build_evidence(tx_val, cycles, top_accounts)

    # Feature importance
    fi = feature_importance(fused, val_labels)

    # Persist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(metrics.items(), columns=["metric", "value"]).to_csv(REPORTS_DIR / "eval_metrics.csv", index=False)
    pd.DataFrame([{"account_id": k, "evidence": v} for k, v in evidence.items()]).to_csv(REPORTS_DIR / "eval_evidence.csv", index=False)
    fi.to_csv(REPORTS_DIR / "fusion_feature_importance.csv", index=False)

    print("Saved metrics to reports/eval_metrics.json and eval_metrics.csv")
    print("Saved evidence to reports/eval_evidence.csv (top 50 fused accounts)")
    print("Saved feature importance to reports/fusion_feature_importance.csv")


if __name__ == "__main__":
    main()
