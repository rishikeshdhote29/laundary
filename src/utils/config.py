from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from typing import List

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
FEATURES_DIR: Final[Path] = OUTPUTS_DIR / "features"
LOGS_DIR: Final[Path] = ROOT / "logs"
SCRIPTS_DIR: Final[Path] = ROOT / "scripts"

# Allowed values and labels
ALLOWED_LABELS = {"normal", "structuring", "circular", "layering"}
ALLOWED_CHANNELS = [
    "online_banking",
    "branch_cash",
    "atm",
    "mobile_wallet",
    "crypto_exchange",
]
ALLOWED_CURRENCIES = ["USD", "EUR", "GBP", "INR", "SGD"]

# Canonical split fractions (time-ordered)
TRAIN_FRACTION: Final[float] = 0.70
VAL_FRACTION: Final[float] = 0.15
TEST_FRACTION: Final[float] = 0.15
FRACTION_TOLERANCE: Final[float] = 0.05

# Feature engineering defaults
ROLLING_WINDOWS_HOURS: Final[List[int]] = [24, 72]
SMALL_TX_THRESHOLD: Final[float] = 1000.0
SEQUENCE_N: Final[int] = 10

# Detector thresholds
STRUCTURING_CONTAMINATION: Final[float] = 0.02  # expected anomaly ratio
LAYERING_COUNT_24H_THRESHOLD: Final[int] = 20
LAYERING_COUNT_72H_THRESHOLD: Final[int] = 50
LAYERING_MEAN_GAP_24H_SEC: Final[float] = 1800.0  # 30 minutes
LAYERING_OUT_DEG_THRESHOLD: Final[int] = 10

# Fusion weights (weighted average fallback)
FUSION_WEIGHTS: Final[dict] = {
    "structuring": 0.4,
    "circular": 0.3,
    "layering": 0.3,
}

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
        FEATURES_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
