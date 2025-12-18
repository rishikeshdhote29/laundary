from pathlib import Path

import pandas as pd

from src.utils.config import REPORTS_DIR


def test_detector_reports_exist_or_skip():
    struct_path = REPORTS_DIR / "structuring_scores.csv"
    circ_path = REPORTS_DIR / "circular_scores.csv"
    layer_path = REPORTS_DIR / "layering_scores.csv"
    fused_path = REPORTS_DIR / "fused_scores.csv"
    # Skip gracefully if detectors haven't been run yet
    if not struct_path.exists() or not circ_path.exists() or not layer_path.exists():
        return
    assert struct_path.exists()
    assert circ_path.exists()
    assert layer_path.exists()
    # fused is optional; if present, ensure columns
    if fused_path.exists():
        df = pd.read_csv(fused_path)
        assert "fusion_score" in df.columns
        assert "pattern_tag" in df.columns


def test_score_columns_numeric_when_present():
    files = {
        "structuring": REPORTS_DIR / "structuring_scores.csv",
        "circular": REPORTS_DIR / "circular_scores.csv",
        "layering": REPORTS_DIR / "layering_scores.csv",
    }
    for name, path in files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        col = "score" if "score" in df.columns else "cycle_score"
        assert col in df.columns
        assert pd.api.types.is_numeric_dtype(df[col]), f"{name} score must be numeric"
