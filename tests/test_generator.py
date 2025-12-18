from pathlib import Path

import pandas as pd

from src.utils.config import RAW_DIR, TRANSACTIONS_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV
from src.utils.validation import run_sanity_checks


def test_dataset_files_and_splits_present():
    # If the dataset hasn't been generated yet, skip this test gracefully.
    if not TRANSACTIONS_CSV.exists():
        return

    assert TRAIN_CSV.exists()
    assert VAL_CSV.exists()
    assert TEST_CSV.exists()


def test_sanity_checks_pass():
    # Skip if dataset is missing
    if not TRANSACTIONS_CSV.exists():
        return
    report = run_sanity_checks(RAW_DIR)
    assert report.passed, f"Sanity checks failed: {report.details}"
