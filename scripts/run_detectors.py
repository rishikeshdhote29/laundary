#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Ensure project root for src imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import (
    RAW_DIR,
    FEATURES_DIR,
    REPORTS_DIR,
    ensure_dirs,
    STRUCTURING_CONTAMINATION,
    LAYERING_COUNT_24H_THRESHOLD,
    LAYERING_COUNT_72H_THRESHOLD,
    LAYERING_MEAN_GAP_24H_SEC,
    LAYERING_OUT_DEG_THRESHOLD,
)


def load_features() -> Dict[str, pd.DataFrame]:
    tx = pd.read_csv(RAW_DIR / "transaction.csv", parse_dates=["timestamp"])
    txn_feats = pd.read_csv(FEATURES_DIR / "transaction_features.csv", parse_dates=["timestamp"]) if (FEATURES_DIR / "transaction_features.csv").exists() else pd.DataFrame()
    roll = pd.read_csv(FEATURES_DIR / "transaction_account_rolling.csv", parse_dates=["timestamp"]) if (FEATURES_DIR / "transaction_account_rolling.csv").exists() else pd.DataFrame()
    graph = pd.read_csv(FEATURES_DIR / "account_graph_features.csv") if (FEATURES_DIR / "account_graph_features.csv").exists() else pd.DataFrame()
    seq = pd.read_csv(FEATURES_DIR / "account_sequence_features.csv") if (FEATURES_DIR / "account_sequence_features.csv").exists() else pd.DataFrame()
    return {"tx": tx, "txn_feats": txn_feats, "roll": roll, "graph": graph, "seq": seq}


def structuring_detector(roll: pd.DataFrame) -> pd.DataFrame:
    """Score accounts using IsolationForest on receiver rolling features."""
    if roll.empty:
        return pd.DataFrame(columns=["account_id", "score"]) 

    # Use receiver-side 24/72h features
    cols = [
        "receiver_count_24h", "receiver_sum_24h", "receiver_median_24h", "receiver_std_24h", "receiver_small_ratio_24h",
        "receiver_count_72h", "receiver_sum_72h", "receiver_median_72h", "receiver_std_72h", "receiver_small_ratio_72h",
    ]
    use = roll[["receiver_id", "timestamp"] + [c for c in cols if c in roll.columns]].copy()
    use = use.dropna()
    # Take the latest record per account
    use = use.sort_values(["receiver_id", "timestamp"]).groupby("receiver_id").tail(1)
    X = use.drop(columns=["receiver_id", "timestamp"]).values

    try:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(n_estimators=200, contamination=STRUCTURING_CONTAMINATION, random_state=42)
        iso.fit(X)
        scores = -iso.decision_function(X)  # higher is more anomalous
    except Exception:
        # Fallback: z-score on sum_24h vs count_24h emphasizing many small deposits
        import numpy as np
        s24 = use.get("receiver_sum_24h", pd.Series(0)).to_numpy()
        c24 = use.get("receiver_count_24h", pd.Series(0)).to_numpy()
        small = use.get("receiver_small_ratio_24h", pd.Series(0)).to_numpy()
        x = (np.log1p(s24) * (small + 1.0)) / (np.sqrt(c24 + 1.0))
        m, sd = float(x.mean()), float(x.std() + 1e-9)
        scores = (x - m) / sd

    out = pd.DataFrame({"account_id": use["receiver_id"].values, "score": scores})
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


def circular_detector(tx: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[str]]]:
    """Find small cycles and compute cycle score per node."""
    try:
        import networkx as nx
    except Exception:
        # Fallback: mutual transfer pairs
        pairs = tx.groupby(["sender_id", "receiver_id"]).size().reset_index(name="cnt")
        rev = pairs.merge(pairs, left_on=["sender_id", "receiver_id"], right_on=["receiver_id", "sender_id"], suffixes=("", "_rev"))
        accounts = pd.unique(pd.concat([rev["sender_id"], rev["receiver_id"]]))
        df = pd.DataFrame({"account_id": accounts})
        df["cycle_score"] = 1.0
        return df.sort_values("cycle_score", ascending=False), []

    agg = tx.groupby(["sender_id", "receiver_id"]).size().reset_index(name="cnt")
    G = nx.DiGraph()
    for _, r in agg.iterrows():
        G.add_edge(r["sender_id"], r["receiver_id"]) 

    cycles_paths: List[List[str]] = []
    cycle_counts: Dict[str, int] = {}
    limit_cycles = 500
    for cyc in nx.simple_cycles(G):
        if 3 <= len(cyc) <= 6:
            cycles_paths.append(cyc)
            for n in cyc:
                cycle_counts[n] = cycle_counts.get(n, 0) + 1
            if len(cycles_paths) >= limit_cycles:
                break
    accounts = list(G.nodes())
    df = pd.DataFrame({"account_id": accounts, "cycle_score": [cycle_counts.get(a, 0) for a in accounts]})
    df = df.sort_values("cycle_score", ascending=False)
    return df, cycles_paths


def layering_detector(tx: pd.DataFrame, roll: pd.DataFrame, graph: pd.DataFrame, seq: pd.DataFrame) -> pd.DataFrame:
    """Rule-based layering detector with optional IsolationForest on sequences."""
    # Rule-based flags
    # Latest sender rolling per account
    send_cols = [
        "sender_count_24h", "sender_count_72h", "sender_mean_gap_24h", "sender_mean_gap_72h",
    ]
    send = roll[["sender_id", "timestamp"] + [c for c in send_cols if c in roll.columns]].copy()
    send = send.dropna()
    send_latest = send.sort_values(["sender_id", "timestamp"]).groupby("sender_id").tail(1)

    # Merge with graph degrees
    g = graph[["account_id", "out_degree"]] if not graph.empty else pd.DataFrame(columns=["account_id", "out_degree"])
    rules = send_latest.rename(columns={"sender_id": "account_id"}).merge(g, on="account_id", how="left")
    rules["out_degree"].fillna(0, inplace=True)

    rule_flag = (
        (rules.get("sender_count_24h", 0) > LAYERING_COUNT_24H_THRESHOLD) |
        (rules.get("sender_count_72h", 0) > LAYERING_COUNT_72H_THRESHOLD) |
        (rules.get("sender_mean_gap_24h", LAYERING_MEAN_GAP_24H_SEC + 1) < LAYERING_MEAN_GAP_24H_SEC) |
        (rules["out_degree"] > LAYERING_OUT_DEG_THRESHOLD)
    )
    rules["rule_score"] = rule_flag.astype(int)

    # ML score on sequences if available
    ml_score = pd.DataFrame({"account_id": rules["account_id"], "ml_score": 0.0})
    if not seq.empty:
        try:
            from sklearn.ensemble import IsolationForest
            # Use last amounts and gaps as features
            seq_feats = seq.copy()
            feat_cols = [c for c in seq_feats.columns if c.startswith("last_amount_") or c.startswith("last_gap_")]
            X = seq_feats[feat_cols].values
            iso = IsolationForest(n_estimators=200, contamination=STRUCTURING_CONTAMINATION, random_state=42)
            iso.fit(X)
            scores = -iso.decision_function(X)
            ml_score = pd.DataFrame({"account_id": seq_feats["account_id"].values, "ml_score": scores})
        except Exception:
            pass

    out = rules.merge(ml_score, on="account_id", how="left")
    out["ml_score"].fillna(0.0, inplace=True)
    out["score"] = out["rule_score"] * 0.6 + out["ml_score"] * 0.4
    out = out.sort_values("score", ascending=False)
    return out[["account_id", "score", "rule_score", "ml_score", "out_degree"]]


def main() -> None:
    ensure_dirs()
    feats = load_features()
    tx, roll, graph, seq = feats["tx"], feats["roll"], feats["graph"], feats["seq"]

    # Structuring
    struct_scores = structuring_detector(roll)

    # Circular
    circ_scores, cycles_paths = circular_detector(tx)

    # Layering
    layer_scores = layering_detector(tx, roll, graph, seq)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    struct_scores.to_csv(REPORTS_DIR / "structuring_scores.csv", index=False)
    circ_scores.to_csv(REPORTS_DIR / "circular_scores.csv", index=False)
    layer_scores.to_csv(REPORTS_DIR / "layering_scores.csv", index=False)

    # Top-50 display
    print("Top 50 structuring accounts:")
    print(struct_scores.head(50))
    print("Top 50 circular accounts:")
    print(circ_scores.head(50))
    print("Top 50 layering accounts:")
    print(layer_scores.head(50))

    # Save cycle evidence
    evidence_path = REPORTS_DIR / "circular_cycles_evidence.txt"
    with open(evidence_path, "w", encoding="utf-8") as f:
        for cyc in cycles_paths[:200]:
            f.write("->".join(cyc) + "\n")
    print(f"Saved cycle evidence to {evidence_path}")


if __name__ == "__main__":
    main()
