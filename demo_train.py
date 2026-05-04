#!/usr/bin/env python3
"""Train and compare model performance"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

from train import create_training_data
from yield_calc.data import YieldConfig, FeatureEngineer
from yield_calc.modules import YieldNet, YieldNetWithAttention
from yield_calc.tools import Trainer, TrainingConfig, set_random_seed

print("=" * 70)
print("YIELD CALCULATOR - TRAINING & PERFORMANCE COMPARISON")
print("=" * 70)

set_random_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create training data
print("\n[1] Generating training data (2000 samples)...")
df = create_training_data(num_samples=2000)

# Feature engineering
print("[2] Engineering features...")
config = YieldConfig()
fe = FeatureEngineer(config)
df_eng = fe.engineer_features(df)

X = df_eng.drop('E', axis=1).values
y = df_eng[['E']].values

print(f"    Input dimension: {df_eng.shape[1] - 1}")
print(f"    Output dimension: 1")
print(f"    Total samples: {len(X)}")

# Normalize
scaler_x = StandardScaler()
scaler_y = MinMaxScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)

# Create dataset
dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
train_dataset, val_dataset = random_split(dataset, [1600, 400], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

print(f"    Train samples: {len(train_dataset)}")
print(f"    Val samples: {len(val_dataset)}")

# Model 1: Standard Network
print("\n[3] Training YieldNet (Standard Residual)...")
model1 = YieldNet(
    input_dim=X.shape[1],
    hidden_dim=256,
    num_layers=4,
    dropout=0.15
)

trainer1 = Trainer(model1, TrainingConfig(
    num_epochs=500,
    batch_size=32,
    learning_rate=0.001,
    weight_decay=1e-4,
    patience=50,
    checkpoint_dir="checkpoints",
    device=device,
))

print("    Training... (this may take a minute)")
history1 = trainer1.train(train_loader, val_loader)

final_train_loss_1 = history1["train_loss"][-1]
final_val_loss_1 = history1["val_loss"][-1]
final_val_mae_1 = history1["val_metrics"][-1]["mae"]
final_val_r2_1 = history1["val_metrics"][-1]["r2"]

# Model 2: Attention Network (MACE-inspired)
print("\n[4] Training YieldNetWithAttention (Transformer-based)...")
model2 = YieldNetWithAttention(
    input_dim=X.shape[1],
    hidden_dim=256,
    num_layers=4,
    num_heads=8,
    dropout=0.15
)

trainer2 = Trainer(model2, TrainingConfig(
    num_epochs=500,
    batch_size=32,
    learning_rate=0.001,
    weight_decay=1e-4,
    patience=50,
    checkpoint_dir="checkpoints",
    device=device,
))

print("    Training... (this may take a minute)")
history2 = trainer2.train(train_loader, val_loader)

final_train_loss_2 = history2["train_loss"][-1]
final_val_loss_2 = history2["val_loss"][-1]
final_val_mae_2 = history2["val_metrics"][-1]["mae"]
final_val_r2_2 = history2["val_metrics"][-1]["r2"]

# Performance Comparison
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

print(f"\n{'Metric':<30} {'YieldNet':<20} {'Attention':<20} {'Improvement'}")
print("-" * 70)

def format_metric(val):
    return f"{val:.6f}"

# Training Loss
improvement_train = ((final_train_loss_1 - final_train_loss_2) / max(final_train_loss_1, 1e-6)) * 100
print(f"{'Final Training Loss':<30} {format_metric(final_train_loss_1):<20} {format_metric(final_train_loss_2):<20} {improvement_train:+.2f}%")

# Validation Loss
improvement_val = ((final_val_loss_1 - final_val_loss_2) / max(final_val_loss_1, 1e-6)) * 100
print(f"{'Final Validation Loss':<30} {format_metric(final_val_loss_1):<20} {format_metric(final_val_loss_2):<20} {improvement_val:+.2f}%")

# MAE
improvement_mae = ((final_val_mae_1 - final_val_mae_2) / max(final_val_mae_1, 1e-6)) * 100
print(f"{'Validation MAE':<30} {format_metric(final_val_mae_1):<20} {format_metric(final_val_mae_2):<20} {improvement_mae:+.2f}%")

# R²
improvement_r2 = ((final_val_r2_2 - final_val_r2_1) / max(abs(final_val_r2_1), 1e-6)) * 100
print(f"{'Validation R²':<30} {format_metric(final_val_r2_1):<20} {format_metric(final_val_r2_2):<20} {improvement_r2:+.2f}%")

print("\n" + "=" * 70)
print("TRAINING EFFICIENCY")
print("=" * 70)

epochs_trained_1 = len(history1["train_loss"])
epochs_trained_2 = len(history2["train_loss"])

print(f"YieldNet (Standard):       {epochs_trained_1} epochs")
print(f"YieldNetWithAttention:     {epochs_trained_2} epochs")

# Count parameters
params1 = sum(p.numel() for p in model1.parameters())
params2 = sum(p.numel() for p in model2.parameters())

print(f"\nModel Parameters:")
print(f"YieldNet (Standard):       {params1:,} parameters")
print(f"YieldNetWithAttention:     {params2:,} parameters")

# Save models
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

os.makedirs("checkpoints", exist_ok=True)

model1_path = "checkpoints/yield_model_standard.pt"
torch.save({
    "model_state": model1.state_dict(),
    "config": config,
    "model_type": "standard",
    "input_dim": X.shape[1],
}, model1_path)
print(f"✓ Saved: {model1_path}")

model2_path = "checkpoints/yield_model_attention.pt"
torch.save({
    "model_state": model2.state_dict(),
    "config": config,
    "model_type": "attention",
    "input_dim": X.shape[1],
}, model2_path)
print(f"✓ Saved: {model2_path}")

# Save scalers
import joblib
joblib.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, model1_path.replace(".pt", "_scalers.joblib"))
joblib.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, model2_path.replace(".pt", "_scalers.joblib"))

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print(f"1. The attention-based model ({final_val_mae_2:.4f} MAE) is recommended for best performance")
print("2. Run: python main.py")
print("=" * 70 + "\n")
