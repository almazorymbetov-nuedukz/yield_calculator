"""Simple HPO script: runs several short training runs and records validation MAE.

This imports `train_model` from `train.py` and calls it with different argparse.Namespace configs.
"""
import json
import os
from types import SimpleNamespace
from pathlib import Path
import pandas as pd

from train import train_model

ROOT = Path('.')
TRAIN_CSV = ROOT / 'data' / 'training_data.csv'

# Small set of hyperparameter configurations (short runs)
configs = [
    {"lr": 1e-3, "hidden": 128, "dropout": 0.1, "weight_decay": 1e-4, "batch": 64, "epochs": 30},
    {"lr": 1e-4, "hidden": 256, "dropout": 0.15, "weight_decay": 1e-4, "batch": 64, "epochs": 30},
    {"lr": 5e-4, "hidden": 256, "dropout": 0.2, "weight_decay": 1e-5, "batch": 32, "epochs": 30},
    {"lr": 1e-4, "hidden": 128, "dropout": 0.1, "weight_decay": 1e-5, "batch": 32, "epochs": 30},
]

results = []

for i, cfg in enumerate(configs, start=1):
    print(f"Running trial {i}/{len(configs)}: {cfg}")
    out_dir = ROOT / f"checkpoints/hpo_trial_{i}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build args namespace compatible with train.py
    args = SimpleNamespace(
        train_file=str(TRAIN_CSV),
        num_samples=8000,
        model_type='attention',
        hidden_dim=cfg['hidden'],
        num_layers=4,
        attention_heads=8,
        dropout=cfg['dropout'],
        num_epochs=cfg['epochs'],
        batch_size=cfg['batch'],
        learning_rate=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        patience=30,
        device='cpu',
        dtype='float32',
        seed=42,
        checkpoint_dir=str(out_dir)
    )

    # Run training (short)
    try:
        train_model(args)
    except Exception as e:
        print(f"Trial {i} failed: {e}")
        results.append({"trial": i, "error": str(e), **cfg})
        continue

    # Read training history if present
    history_path = out_dir / 'training_history.json'
    best_val_mae = None
    if history_path.exists():
        with open(history_path, 'r') as f:
            hist = json.load(f)
        # Extract minimum val MAE from recorded val_metrics
        val_metrics = hist.get('val_metrics', [])
        maes = [m.get('mae') for m in val_metrics if isinstance(m, dict) and 'mae' in m]
        if maes:
            best_val_mae = min(maes)

    results.append({"trial": i, "checkpoint": str(out_dir), "best_val_mae": best_val_mae, **cfg})

# Save summary
summary_path = ROOT / 'checkpoints' / 'hpo_summary.json'
summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2)

# Also produce CSV
csv_path = ROOT / 'checkpoints' / 'hpo_summary.csv'
pd.DataFrame(results).to_csv(csv_path, index=False)

print('HPO complete. Results written to:', summary_path)
