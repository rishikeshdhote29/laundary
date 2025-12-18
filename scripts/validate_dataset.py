#!/usr/bin/env python3
from __future__ import annotations

from pprint import pprint
import sys
from pathlib import Path

# Ensure project root is on sys.path for 'src' imports when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import RAW_DIR, ensure_dirs
from src.utils.validation import run_sanity_checks


def main() -> None:
    ensure_dirs()
    report = run_sanity_checks(RAW_DIR)
    print("Sanity checks passed:" if report.passed else "Sanity checks failed!")
    pprint(report.checks)
    pprint(report.details)


if __name__ == "__main__":
    main()
