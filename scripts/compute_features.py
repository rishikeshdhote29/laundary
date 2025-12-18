#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd

# Ensure project root is on sys.path for 'src' imports when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import (
    RAW_DIR,
    FEATURES_DIR,
    ensure_dirs,
    ROLLING_WINDOWS_HOURS,
    SMALL_TX_THRESHOLD,
    SEQUENCE_N,
)


def load_raw(raw_dir: Path = RAW_DIR) -> Dict[str, pd.DataFrame]:
    tx = pd.read_csv(raw_dir / "transaction.csv", parse_dates=["timestamp"])
    accounts = pd.read_csv(raw_dir / "accounts.csv", parse_dates=["created_at"]) if (raw_dir / "accounts.csv").exists() else pd.DataFrame()
    return {"tx": tx, "accounts": accounts}


def predominant_country_per_account(tx: pd.DataFrame) -> pd.DataFrame:
    # Count appearances of account as sender/receiver by country and pick argmax
    send = tx.groupby(["sender_id", "country"]).size().reset_index(name="cnt")
    recv = tx.groupby(["receiver_id", "country"]).size().reset_index(name="cnt")
    send.rename(columns={"sender_id": "account_id"}, inplace=True)
    recv.rename(columns={"receiver_id": "account_id"}, inplace=True)
    comb = pd.concat([send, recv], ignore_index=True)
    idx = comb.groupby("account_id")[["cnt"]].idxmax()
    rows = comb.loc[idx["cnt"], ["account_id", "country"]]
    return rows.rename(columns={"country": "pred_country"})


def transaction_level_features(tx: pd.DataFrame, acc_country: pd.DataFrame) -> pd.DataFrame:
    df = tx.copy()
    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.weekday >= 5
    # Simple channel flag: crypto channel indicator
    df["channel_flag"] = (df["channel"] == "crypto_exchange").astype(int)

    # Sender/receiver predominant countries and country_pair
    acc_country = acc_country.set_index("account_id")
    df["sender_country"] = df["sender_id"].map(acc_country["pred_country"]) 
    df["receiver_country"] = df["receiver_id"].map(acc_country["pred_country"]) 
    df["country_pair"] = (df["sender_country"].fillna("?") + "->" + df["receiver_country"].fillna("?"))

    return df[[
        "tx_id","timestamp","sender_id","receiver_id","amount","currency","channel","country","tx_type","label","scenario_id","evidence",
        "log_amount","hour_of_day","is_weekend","channel_flag","sender_country","receiver_country","country_pair"
    ]]


def _rolling_by_account(df: pd.DataFrame, account_col: str, ts_col: str, val_col: str, windows_hours: List[int]) -> pd.DataFrame:
    work = df[[account_col, ts_col, val_col]].copy()
    work = work.sort_values([account_col, ts_col])
    # Precompute interarrival as seconds per account
    work["gap_sec"] = work.groupby(account_col)[ts_col].diff().dt.total_seconds().fillna(0)
    frames = []
    for h in windows_hours:
        window = f"{h}h"
        # Rolling aggregations on amount
        agg_amt = (
            work.groupby(account_col)
            .rolling(window=window, on=ts_col)[val_col]
            .agg(["count", "sum", "median", "std"])
            .reset_index()
        )
        # Rolling ratio of small transactions
        small_series = (work[val_col] < SMALL_TX_THRESHOLD).astype(float)
        work_small = work.copy()
        work_small["is_small"] = small_series
        agg_small = (
            work_small.groupby(account_col)
            .rolling(window=window, on=ts_col)["is_small"]
            .mean()
            .reset_index()
        )
        # Rolling mean gap seconds
        agg_gap = (
            work.groupby(account_col)
            .rolling(window=window, on=ts_col)["gap_sec"]
            .mean()
            .reset_index()
        )
        # Merge these for the window
        agg = agg_amt.merge(agg_small, on=[account_col, ts_col], how="left").merge(
            agg_gap, on=[account_col, ts_col], how="left"
        )
        # Align with original row order by adding an index
        agg.rename(columns={
            "count": f"count_{h}h",
            "sum": f"sum_{h}h",
            "median": f"median_{h}h",
            "std": f"std_{h}h",
            "is_small": f"small_ratio_{h}h",
            "gap_sec": f"mean_gap_{h}h",
        }, inplace=True)
        frames.append(agg)
    # Merge frames on account+timestamp
    merged = frames[0]
    for fr in frames[1:]:
        merged = merged.merge(fr, on=[account_col, ts_col], how="left")
    return merged


def account_window_features(tx: pd.DataFrame) -> pd.DataFrame:
    # Sender-side rolling
    send_df = tx[["sender_id", "timestamp", "amount"]].rename(columns={"sender_id": "account_id"})
    send_roll = _rolling_by_account(send_df, "account_id", "timestamp", "amount", ROLLING_WINDOWS_HOURS)
    send_roll = send_roll.add_prefix("sender_")
    send_roll.rename(columns={"sender_account_id": "sender_id", "sender_timestamp": "timestamp"}, inplace=True)

    # Receiver-side rolling
    recv_df = tx[["receiver_id", "timestamp", "amount"]].rename(columns={"receiver_id": "account_id"})
    recv_roll = _rolling_by_account(recv_df, "account_id", "timestamp", "amount", ROLLING_WINDOWS_HOURS)
    recv_roll = recv_roll.add_prefix("receiver_")
    recv_roll.rename(columns={"receiver_account_id": "receiver_id", "receiver_timestamp": "timestamp"}, inplace=True)

    # Join back to transactions by sender_id/timestamp and receiver_id/timestamp
    base = tx[["tx_id", "timestamp", "sender_id", "receiver_id"]].copy()
    out = base.merge(send_roll, on=["sender_id", "timestamp"], how="left")
    out = out.merge(recv_roll, on=["receiver_id", "timestamp"], how="left")
    return out


def graph_features(tx: pd.DataFrame) -> pd.DataFrame:
    try:
        import networkx as nx
    except ImportError:
        # Fallback: degrees from aggregated edges; other metrics as zeros
        agg = (
            tx.groupby(["sender_id", "receiver_id"]).agg(tx_count=("tx_id", "count"), total_amount=("amount", "sum")).reset_index()
        )
        out_deg = agg.groupby("sender_id").size().rename("out_degree")
        in_deg = agg.groupby("receiver_id").size().rename("in_degree")
        accounts = set(agg["sender_id"]).union(set(agg["receiver_id"]))
        df = pd.DataFrame({
            "account_id": list(accounts),
        })
        df["in_degree"] = df["account_id"].map(in_deg).fillna(0).astype(int)
        df["out_degree"] = df["account_id"].map(out_deg).fillna(0).astype(int)
        df["pagerank"] = 0.0
        df["clustering"] = 0.0
        df["cycle_presence"] = 0
        return df

    # Aggregate edges with total_amount as weight
    agg = (
        tx.groupby(["sender_id", "receiver_id"]).agg(tx_count=("tx_id", "count"), total_amount=("amount", "sum")).reset_index()
    )
    G = nx.DiGraph()
    for _, r in agg.iterrows():
        G.add_edge(r["sender_id"], r["receiver_id"], weight=float(r["total_amount"]))
    nodes = list(G.nodes())

    # Degrees
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # PageRank (weighted)
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {n: 0.0 for n in nodes}

    # Clustering via undirected projection
    und = G.to_undirected()
    clustering = nx.clustering(und)

    # Cycle presence via SCC size > 1
    scc = list(nx.strongly_connected_components(G))
    scc_size: Dict[str, int] = {}
    for comp in scc:
        size = len(comp)
        for n in comp:
            scc_size[n] = size
    cycle_presence = {n: (scc_size.get(n, 1) > 1) for n in nodes}

    df = pd.DataFrame({
        "account_id": nodes,
        "in_degree": [in_deg.get(n, 0) for n in nodes],
        "out_degree": [out_deg.get(n, 0) for n in nodes],
        "pagerank": [pr.get(n, 0.0) for n in nodes],
        "clustering": [clustering.get(n, 0.0) for n in nodes],
        "cycle_presence": [int(cycle_presence.get(n, False)) for n in nodes],
    })
    return df


def sequence_features(tx: pd.DataFrame) -> pd.DataFrame:
    # Outgoing sequences per account
    tx = tx.sort_values(["sender_id", "timestamp"]).copy()
    feats = []
    for acc, grp in tx.groupby("sender_id"):
        amounts = grp["amount"].tolist()
        times = grp["timestamp"].tolist()
        gaps = [0.0] + [float((times[i] - times[i-1]).total_seconds()) for i in range(1, len(times))]
        # Take last N
        last_amt = amounts[-SEQUENCE_N:]
        last_gap = gaps[-SEQUENCE_N:]
        # Pad to N with zeros
        la = ([0.0] * (SEQUENCE_N - len(last_amt))) + last_amt
        lg = ([0.0] * (SEQUENCE_N - len(last_gap))) + last_gap
        feat = {
            "account_id": acc,
            "seq_len_norm": min(len(amounts) / SEQUENCE_N, 1.0),
        }
        for i in range(SEQUENCE_N):
            feat[f"last_amount_{i+1}"] = la[i]
            feat[f"last_gap_{i+1}"] = lg[i]
        feats.append(feat)
    return pd.DataFrame(feats)


def main() -> None:
    ensure_dirs()
    raw = load_raw()
    tx = raw["tx"]

    acc_country = predominant_country_per_account(tx)
    txn_feats = transaction_level_features(tx, acc_country)
    acct_roll = account_window_features(tx)
    g_feats = graph_features(tx)
    seq_feats = sequence_features(tx)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    txn_feats.to_csv(FEATURES_DIR / "transaction_features.csv", index=False)
    acct_roll.to_csv(FEATURES_DIR / "transaction_account_rolling.csv", index=False)
    g_feats.to_csv(FEATURES_DIR / "account_graph_features.csv", index=False)
    seq_feats.to_csv(FEATURES_DIR / "account_sequence_features.csv", index=False)
    print("Feature tables written:")
    print(FEATURES_DIR / "transaction_features.csv")
    print(FEATURES_DIR / "transaction_account_rolling.csv")
    print(FEATURES_DIR / "account_graph_features.csv")
    print(FEATURES_DIR / "account_sequence_features.csv")


if __name__ == "__main__":
    main()
