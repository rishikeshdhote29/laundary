from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Project root inferred from src/utils/config.py -> ../../
ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# Core directories
DATA_DIR: Final[Path] = ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
MODELS_DIR: Final[Path] = ROOT / "models"
ONNX_DIR: Final[Path] = MODELS_DIR / "onnx"
PYTORCH_DIR: Final[Path] = MODELS_DIR / "pytorch"
SKLEARN_DIR: Final[Path] = MODELS_DIR / "sklearn"
OUTPUTS_DIR: Final[Path] = ROOT / "outputs"
FIGURES_DIR: Final[Path] = OUTPUTS_DIR / "figures"
REPORTS_DIR: Final[Path] = OUTPUTS_DIR / "reports"
LOGS_DIR: Final[Path] = ROOT / "logs"
SCRIPTS_DIR: Final[Path] = ROOT / "scripts"

# Canonical raw dataset file paths
TRANSACTIONS_CSV: Final[Path] = RAW_DIR / "transaction.csv"
ACCOUNTS_CSV: Final[Path] = RAW_DIR / "accounts.csv"
EDGES_CSV: Final[Path] = RAW_DIR / "edges.csv"
TRAIN_CSV: Final[Path] = RAW_DIR / "train.csv"
VAL_CSV: Final[Path] = RAW_DIR / "val.csv"
TEST_CSV: Final[Path] = RAW_DIR / "test.csv"


@dataclass(frozen=True)
class DatasetSizes:
    """Default dataset sizes and ratios for generation.

    Adjust these in one place; scripts can import and override via CLI if needed.
    """

    n_accounts: int = 10_000
    n_transactions: int = 100_000
    suspicious_low: float = 0.01
    suspicious_high: float = 0.03
    num_days: int = 120
    seed: int = 42


# Default instance to import directly
DEFAULT_SIZES = DatasetSizes()


def ensure_dirs() -> None:
    """Create frequently used directories if they don't exist."""
    for p in [
        RAW_DIR,
        ONNX_DIR,
        PYTORCH_DIR,
        SKLEARN_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
