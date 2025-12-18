$Root = Split-Path -Parent $PSScriptRoot
python "$Root/scripts/generate_dataset.py" --n-accounts 10000 --n-transactions 100000 --output-dir "$Root/data/raw" --seed 42
