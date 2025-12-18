$Root = Split-Path -Parent $PSScriptRoot
python "$Root/scripts/fuse_scores.py"
