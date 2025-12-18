#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import (
    FEATURES_DIR,
    REPORTS_DIR,
    ONNX_DIR,
    ensure_dirs,
)


def load_structuring_training() -> Tuple[np.ndarray, np.ndarray, list]:
    roll_path = FEATURES_DIR / "transaction_account_rolling.csv"
    if not roll_path.exists():
        raise FileNotFoundError("Missing rolling features. Run scripts/compute_features.py first.")
    roll = pd.read_csv(roll_path, parse_dates=["timestamp"])
    cols = [
        "receiver_count_24h", "receiver_sum_24h", "receiver_median_24h", "receiver_std_24h", "receiver_small_ratio_24h",
        "receiver_count_72h", "receiver_sum_72h", "receiver_median_72h", "receiver_std_72h", "receiver_small_ratio_72h",
    ]
    use_cols = [c for c in cols if c in roll.columns]
    df = roll[["receiver_id", "timestamp"] + use_cols].dropna()
    df = df.sort_values(["receiver_id", "timestamp"]).groupby("receiver_id").tail(1)
    X = df[use_cols].to_numpy(dtype=np.float32)
    feature_names = use_cols
    # Unsupervised: labels all zeros
    y = np.zeros(len(df), dtype=np.int64)
    return X, y, feature_names


def load_fusion_training() -> Tuple[np.ndarray, np.ndarray, list]:
    fused_path = REPORTS_DIR / "fused_scores.csv"
    val_path = Path(ROOT) / "data" / "raw" / "val.csv"
    if not fused_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing fused_scores.csv or val.csv. Run fusion and ensure val split exists.")
    fused = pd.read_csv(fused_path)
    val = pd.read_csv(val_path)
    parts = []
    for col in ["sender_id", "receiver_id"]:
        sub = val[[col, "label"]].rename(columns={col: "account_id"})
        parts.append(sub)
    acc = pd.concat(parts, ignore_index=True)
    acc["y"] = (acc["label"] != "normal").astype(int)
    labels = acc.groupby("account_id")["y"].max().reset_index()
    merged = fused.merge(labels, on="account_id", how="inner")
    if merged.empty or merged["y"].sum() == 0:
        raise ValueError("Not enough positive labels in val set for fusion training.")
    X = merged[["structuring_score", "circular_score", "layering_score"]].to_numpy(dtype=np.float32)
    y = merged["y"].astype(np.int64).to_numpy()
    return X, y, ["structuring_score", "circular_score", "layering_score"]


def export_isoforest(X: np.ndarray, feature_names: list, out_path: Path) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    model.fit(X)

    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset={"": 12, "ai.onnx.ml": 3},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Validate with onnxruntime
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    got = np.array(sess.run(None, {input_name: X[:8].astype(np.float32)})[0]).ravel()
    skl_scores = -model.decision_function(X[:8])
    # Some converters emit scores with opposite sign; check both
    ok = False
    if np.allclose(got, skl_scores, atol=1e-3, rtol=1e-2):
        ok = True
    elif np.allclose(got, -skl_scores, atol=1e-3, rtol=1e-2):
        ok = True
    if not ok:
        print("[warn] ONNX IsolationForest outputs differ from sklearn; proceeding anyway.")


def export_logreg(X: np.ndarray, y: np.ndarray, feature_names: list, out_path: Path) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset={"": 12, "ai.onnx.ml": 3},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Validate
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X[:8].astype(np.float32)})
    # Try: first output is label, second is probabilities (common pattern)
    probs = None
    if len(outputs) >= 2:
        second = np.array(outputs[1])
        if second.ndim == 2 and second.shape[1] == 2:
            probs = second[:, 1]
    if probs is None:
        # Fallback: search for 2-class tensor
        for out in outputs:
            arr = np.array(out)
            if arr.ndim == 2 and arr.shape[1] == 2:
                probs = arr[:, 1]
                break
    if probs is None:
        # Final fallback: use last output as flat array
        probs = np.array(outputs[-1]).ravel()
    got = np.array(probs).ravel()
    skl_scores = model.predict_proba(X[:8])[:, 1]
    if not np.allclose(got, skl_scores, atol=1e-3, rtol=1e-2):
        print("[warn] ONNX LogisticRegression outputs differ from sklearn; proceeding anyway.")


def main() -> None:
    ensure_dirs()
    reports = []

    # Structuring IsolationForest export
    try:
        X_iso, _, fn_iso = load_structuring_training()
        iso_path = ONNX_DIR / "structuring_isoforest.onnx"
        export_isoforest(X_iso, fn_iso, iso_path)
        reports.append(f"Exported {iso_path}")
    except Exception as e:
        reports.append(f"Structuring export skipped: {e}")

    # Fusion Logistic Regression export
    try:
        X_fus, y_fus, fn_fus = load_fusion_training()
        lr_path = ONNX_DIR / "fusion_logreg.onnx"
        export_logreg(X_fus, y_fus, fn_fus, lr_path)
        reports.append(f"Exported {lr_path}")
    except Exception as e:
        reports.append(f"Fusion export skipped: {e}")

    for r in reports:
        print(r)


if __name__ == "__main__":
    main()
