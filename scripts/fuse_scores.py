#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# Ensure project root import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import RAW_DIR, REPORTS_DIR, FUSION_WEIGHTS, ensure_dirs


def load_reports() -> Dict[str, pd.DataFrame]:
    reports = {}
    paths = {
        "structuring": REPORTS_DIR / "structuring_scores.csv",
        "circular": REPORTS_DIR / "circular_scores.csv",
        "layering": REPORTS_DIR / "layering_scores.csv",
    }
    for name, path in paths.items():
        if path.exists():
            df = pd.read_csv(path)
            # normalize column names
            if "account_id" not in df.columns:
                continue
            if "score" not in df.columns:
                # for circular, may be cycle_score
                if "cycle_score" in df.columns:
                    df = df.rename(columns={"cycle_score": "score"})
                else:
                    continue
            reports[name] = df[["account_id", "score"]].copy()
        else:
            reports[name] = pd.DataFrame(columns=["account_id", "score"])
    return reports


def load_val_labels() -> pd.DataFrame:
    val_path = RAW_DIR / "val.csv"
    if not val_path.exists():
        return pd.DataFrame(columns=["account_id", "label"])
    val = pd.read_csv(val_path)
    # Account-level label: suspicious if account participates in any non-normal tx
    val_accounts = pd.concat([val[["sender_id", "label"]].rename(columns={"sender_id": "account_id"}),
                              val[["receiver_id", "label"]].rename(columns={"receiver_id": "account_id"})])
    val_accounts["y"] = (val_accounts["label"] != "normal").astype(int)
    agg = val_accounts.groupby("account_id")["y"].max().reset_index()
    return agg


def weighted_fusion(structuring: pd.DataFrame, circular: pd.DataFrame, layering: pd.DataFrame) -> pd.DataFrame:
    # Union of accounts
    accounts = pd.unique(pd.concat([
        structuring.get("account_id", pd.Series(dtype=str)),
        circular.get("account_id", pd.Series(dtype=str)),
        layering.get("account_id", pd.Series(dtype=str)),
    ], ignore_index=True))
    df = pd.DataFrame({"account_id": accounts})
    df = df.merge(structuring.rename(columns={"score": "structuring_score"}), on="account_id", how="left")
    df = df.merge(circular.rename(columns={"score": "circular_score"}), on="account_id", how="left")
    df = df.merge(layering.rename(columns={"score": "layering_score"}), on="account_id", how="left")
    df.fillna(0.0, inplace=True)

    df["fusion_score"] = (
        FUSION_WEIGHTS.get("structuring", 0.33) * df["structuring_score"] +
        FUSION_WEIGHTS.get("circular", 0.33) * df["circular_score"] +
        FUSION_WEIGHTS.get("layering", 0.34) * df["layering_score"]
    )

    # Pattern tag = argmax of detector scores
    def tag_row(row):
        scores = {
            "structuring": row["structuring_score"],
            "circular": row["circular_score"],
            "layering": row["layering_score"],
        }
        return max(scores, key=scores.get)

    df["pattern_tag"] = df.apply(tag_row, axis=1)
    return df.sort_values("fusion_score", ascending=False).reset_index(drop=True)


def logistic_fusion(base: pd.DataFrame, val_labels: pd.DataFrame) -> pd.DataFrame:
    # Train logistic regression on val labels if available and positive examples exist
    if val_labels.empty or val_labels["y"].sum() == 0:
        return base
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return base

    train = base.merge(val_labels, on="account_id", how="inner")
    if train.empty or train["y"].sum() == 0:
        return base

    X = train[["structuring_score", "circular_score", "layering_score"]].values
    y = train["y"].values
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    proba = model.predict_proba(base[["structuring_score", "circular_score", "layering_score"]].values)[:, 1]
    base["fusion_score"] = proba
    # Recompute pattern tag on scores (unchanged) for explainability
    return base.sort_values("fusion_score", ascending=False).reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    reports = load_reports()
    structuring = reports.get("structuring", pd.DataFrame(columns=["account_id", "score"]))
    circular = reports.get("circular", pd.DataFrame(columns=["account_id", "score"]))
    layering = reports.get("layering", pd.DataFrame(columns=["account_id", "score"]))

    fused = weighted_fusion(structuring, circular, layering)

    # Optional logistic regression refinement using val set labels
    val_labels = load_val_labels()
    fused = logistic_fusion(fused, val_labels)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "fused_scores.csv"
    fused.to_csv(out_path, index=False)
    print(f"Wrote fused scores to {out_path}")
    print("Top 50 fused accounts:")
    print(fused.head(50))


if __name__ == "__main__":
    main()
