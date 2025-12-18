$Root = Split-Path -Parent $PSScriptRoot
python "$Root/scripts/validate_dataset.py"
