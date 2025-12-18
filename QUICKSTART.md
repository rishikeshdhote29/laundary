# Quick Setup

## Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

## Run Complete Pipeline
```powershell
python scripts/generate_dataset.py --n-accounts 10000 --n-transactions 100000 --output-dir data/raw
python scripts/compute_features.py
python scripts/run_detectors.py
python scripts/fuse_scores.py
python scripts/evaluate.py
python scripts/export_onnx.py
```

## Outputs
- `data/raw/` - Synthetic data
- `outputs/features/` - Engineered features
- `outputs/reports/` - Scores, metrics, evidence
- `models/onnx/` - Production models
