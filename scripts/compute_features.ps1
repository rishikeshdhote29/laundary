$Root = Split-Path -Parent $PSScriptRoot
python "$Root/scripts/compute_features.py"
