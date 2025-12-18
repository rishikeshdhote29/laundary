#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/scripts/generate_dataset.py" \
	--n-accounts 10000 \
	--n-transactions 100000 \
	--output-dir "$ROOT_DIR/data/raw" \
	--seed 42
