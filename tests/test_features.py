from pathlib import Path

import pandas as pd

from src.utils.config import FEATURES_DIR


def test_feature_tables_exist_or_skip():
    # Skip if features haven't been computed yet.
    expected = [
        FEATURES_DIR / "transaction_features.csv",
        FEATURES_DIR / "transaction_account_rolling.csv",
        FEATURES_DIR / "account_graph_features.csv",
        FEATURES_DIR / "account_sequence_features.csv",
    ]
    if not all(p.exists() for p in expected):
        return
    for p in expected:
        assert p.exists(), f"missing {p}"


def test_transaction_features_schema_minimum():
    p = FEATURES_DIR / "transaction_features.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    required = ["tx_id", "log_amount", "hour_of_day", "is_weekend", "channel_flag", "country_pair"]
    for c in required:
        assert c in df.columns
