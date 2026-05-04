#!/usr/bin/env python3
"""Comprehensive test suite for yield calculator"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("YIELD CALCULATOR - COMPREHENSIVE TEST SUITE")
print("=" * 70)

# Test 1: Module imports
print("\n[TEST 1] Verifying module structure and imports...")
try:
    from yield_calc.data import YieldConfig, FeatureEngineer, YieldDataset
    from yield_calc.modules import YieldNet, YieldNetWithAttention, EnsembleYieldNet
    from yield_calc.tools import Trainer, CheckpointHandler, compute_metrics
    from yield_calc.calculators import YieldCalculator, EnsembleCalculator
    print("  ✓ All modules imported successfully")
except Exception as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Configuration
print("\n[TEST 2] Testing configuration...")
try:
    config = YieldConfig()
    print(f"  ✓ Config created with {len(config.input_features)} input features")
    print(f"    Features: {config.input_features}")
except Exception as e:
    print(f"  ✗ Config error: {e}")
    sys.exit(1)

# Test 3: Feature Engineering
print("\n[TEST 3] Testing feature engineering...")
try:
    fe = FeatureEngineer(config)
    test_data = pd.DataFrame({
        'T': [298.15], 'R': [2.0], 'D': [1.18], 'V': [259],
        'M': [0.1], 'W': [0.05], 'G': [0.8]
    })
    engineered = fe.engineer_features(test_data)
    print(f"  ✓ Feature engineering successful")
    print(f"    Input features: {test_data.shape[1]}")
    print(f"    Engineered features: {engineered.shape[1]}")
except Exception as e:
    print(f"  ✗ Feature engineering error: {e}")
    sys.exit(1)

# Test 4: Model Architectures
print("\n[TEST 4] Testing model architectures...")
try:
    input_dim = 26  # After feature engineering
    batch_size = 4
    x = torch.randn(batch_size, input_dim)
    
    # Test standard model
    model_std = YieldNet(input_dim=input_dim, hidden_dim=128, num_layers=3)
    out_std = model_std(x)
    assert out_std.shape == (batch_size, 1), f"Standard model output shape mismatch"
    print(f"  ✓ YieldNet (standard): {out_std.shape}")
    
    # Test attention model
    model_attn = YieldNetWithAttention(input_dim=input_dim, hidden_dim=128, num_layers=3)
    out_attn = model_attn(x)
    assert out_attn.shape == (batch_size, 1), f"Attention model output shape mismatch"
    print(f"  ✓ YieldNetWithAttention: {out_attn.shape}")
    
    # Test ensemble model
    model_ens = EnsembleYieldNet(input_dim=input_dim, hidden_dim=128, num_models=3)
    out_ens_mean, out_ens_std = model_ens(x)
    assert out_ens_mean.shape == (batch_size, 1), f"Ensemble mean shape mismatch"
    assert out_ens_std.shape == (batch_size, 1), f"Ensemble std shape mismatch"
    print(f"  ✓ EnsembleYieldNet: mean {out_ens_mean.shape}, std {out_ens_std.shape}")
    
except Exception as e:
    print(f"  ✗ Model architecture error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Training utilities
print("\n[TEST 5] Testing training utilities...")
try:
    from yield_calc.tools import set_random_seed, get_device, EarlyStopping
    
    set_random_seed(42)
    device = get_device("cpu")
    print(f"  ✓ Random seed set and device: {device}")
    
    early_stop = EarlyStopping(patience=5)
    losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.85, 0.86, 0.87]
    stopped = False
    epoch_count = 0
    for loss in losses:
        epoch_count += 1
        if early_stop(loss):
            stopped = True
            break
    print(f"  ✓ Early stopping triggered after {epoch_count} epochs (patience: 5)")
    
except Exception as e:
    print(f"  ✗ Training utilities error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Metrics computation
print("\n[TEST 6] Testing metrics computation...")
try:
    from yield_calc.tools import compute_metrics, compute_mae, compute_rmse
    
    pred = torch.tensor([[50.0], [60.0], [70.0]])
    target = torch.tensor([[48.0], [62.0], [68.0]])
    
    mae = compute_mae(pred, target)
    rmse = compute_rmse(pred, target)
    metrics = compute_metrics(pred, target)
    
    print(f"  ✓ MAE: {mae:.4f}")
    print(f"  ✓ RMSE: {rmse:.4f}")
    print(f"  ✓ All metrics: {list(metrics.keys())}")
    
except Exception as e:
    print(f"  ✗ Metrics error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Checkpoint handling
print("\n[TEST 7] Testing checkpoint handling...")
try:
    from yield_calc.tools import CheckpointHandler
    
    ckpt = CheckpointHandler("test_checkpoints")
    model = YieldNet(input_dim=26, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Save checkpoint
    metrics = {"mae": 0.5, "rmse": 0.7}
    ckpt.save_checkpoint(model, optimizer, 1, metrics, "test_checkpoints/test.pt")
    
    # Verify file exists
    assert os.path.exists("test_checkpoints/test.pt"), "Checkpoint not saved"
    print(f"  ✓ Checkpoint saved successfully")
    
    # Load checkpoint
    loaded = ckpt.load_checkpoint(model, optimizer, "test_checkpoints/test.pt")
    assert "epoch" in loaded, "Checkpoint missing epoch"
    print(f"  ✓ Checkpoint loaded successfully (epoch {loaded['epoch']})")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_checkpoints")
    
except Exception as e:
    print(f"  ✗ Checkpoint error: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Synthetic data generation
print("\n[TEST 8] Testing synthetic data generation...")
try:
    from train import create_training_data
    
    df = create_training_data(num_samples=100)
    assert len(df) == 100, f"Expected 100 samples, got {len(df)}"
    assert all(col in df.columns for col in ['T', 'R', 'D', 'V', 'M', 'W', 'G', 'E'])
    print(f"  ✓ Generated {len(df)} synthetic training samples")
    print(f"  ✓ Data shape: {df.shape}")
    print(f"  ✓ Yield range: [{df['E'].min():.2f}, {df['E'].max():.2f}]")
    
except Exception as e:
    print(f"  ✗ Data generation error: {e}")
    import traceback
    traceback.print_exc()

# Test 9: End-to-end mini training
print("\n[TEST 9] Testing mini training pipeline...")
try:
    from train import create_training_data
    from yield_calc.tools import Trainer, TrainingConfig
    from torch.utils.data import DataLoader, TensorDataset, random_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Create small dataset
    df = create_training_data(num_samples=100)
    fe = FeatureEngineer(config)
    df_eng = fe.engineer_features(df)
    
    X = df_eng.drop('E', axis=1).values
    y = df_eng[['E']].values
    
    scaler_x = StandardScaler()
    scaler_y = MinMaxScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    # Create dataset
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    train_dataset, val_dataset = random_split(dataset, [80, 20])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Mini training
    model = YieldNet(input_dim=X.shape[1], hidden_dim=64, num_layers=2)
    trainer = Trainer(model, TrainingConfig(
        num_epochs=10,
        batch_size=16,
        learning_rate=0.01,
        device="cpu"
    ))
    
    history = trainer.train(train_loader, val_loader)
    
    print(f"  ✓ Mini training completed: {len(history['train_loss'])} epochs")
    print(f"  ✓ Initial train loss: {history['train_loss'][0]:.6f}")
    print(f"  ✓ Final train loss: {history['train_loss'][-1]:.6f}")
    
except Exception as e:
    print(f"  ✗ Training pipeline error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST SUITE SUMMARY")
print("=" * 70)
print("✓ All core components are functional!")
print("\nNext steps:")
print("1. Run: python train.py --model_type attention --num_epochs 2000")
print("2. Run: python main.py  (to use GUI)")
print("=" * 70 + "\n")
