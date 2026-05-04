#!/usr/bin/env python3
"""Training script for yield prediction model (MACE-inspired CLI)"""

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import json
import os

from yield_calc.data import YieldConfig, FeatureEngineer, YieldDataset
from yield_calc.modules import YieldNet, YieldNetWithAttention
from yield_calc.tools import (
    Trainer, TrainingConfig, set_random_seed, get_device, get_dtype
)


def create_training_data(num_samples: int = 8000):
    """Create synthetic training data"""
    raw = {
        'T': [298.15, 303.15, 313.15, 333.15, 298.15, 303.15, 333.15, 298.15, 313.15],
        'R': [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 2.0, 1.0, 2.0],
        'D': [1.18, 1.17, 1.16, 1.14, 1.18, 1.17, 1.14, 1.25, 1.16],
        'V': [259, 231, 102, 41, 376, 324, 40, 1200, 1500],
        'M': [0.1, 0.15, 0.1, 0.1, 0.2, 0.1, 0.1, 0.05, 0.1],
        'W': [0.05, 0.05, 0.08, 0.1, 0.05, 0.1, 3.5, 0.05, 5.0],
        'G': [0.8, 0.6, 0.5, 0.44, 0.75, 0.5, 0.6, 0.55, 0.55],
        'E': [99.1, 96.5, 92.1, 88.5, 96.6, 97.2, 45.0, 20.0, 15.0]
    }
    df = pd.DataFrame(raw)
    
    # Data augmentation via interpolation
    aug = []
    np.random.seed(42)
    
    for _ in range(num_samples):
        s = df.sample(n=1).copy().values[0]
        s2 = df.sample(n=1).values[0]
        alpha = np.random.uniform(0, 1)
        synthetic_row = s * alpha + s2 * (1 - alpha)
        
        # Add noise
        synthetic_row[0] += np.random.uniform(-2, 2)
        synthetic_row[3] *= np.random.uniform(0.9, 1.1)
        
        # Apply constraints
        if synthetic_row[5] > 3.0:
            synthetic_row[7] *= 0.4
        if synthetic_row[3] > 800:
            synthetic_row[7] *= 0.6
        
        saturation_index = synthetic_row[6] / (synthetic_row[4] + 1e-4)
        if saturation_index > 12.0:
            synthetic_row[7] *= (12.0 / saturation_index)
        
        synthetic_row[7] = np.clip(synthetic_row[7], 0, 100)
        aug.append(synthetic_row)
    
    return pd.DataFrame(aug, columns=df.columns)


def train_model(args):
    """Main training function"""
    print("=" * 60)
    print("YIELD CALCULATOR - TRAINING SCRIPT")
    print("=" * 60)
    
    # Set random seed
    set_random_seed(args.seed)
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Model type: {args.model_type}")
    print(f"  Num epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Create/load data
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    if args.train_file:
        print(f"Loading training data from: {args.train_file}")
        df = pd.read_csv(args.train_file)
    else:
        print(f"Generating synthetic training data ({args.num_samples} samples)...")
        df = create_training_data(args.num_samples)
        df.to_csv("training_data.csv", index=False)
        print("Saved to: training_data.csv")
    
    # Feature engineering
    config = YieldConfig()
    feature_engineer = FeatureEngineer(config)
    
    print("Engineering features...")
    df_engineered = feature_engineer.engineer_features(df)
    
    # Prepare X and y
    X = df_engineered.drop('E', axis=1).values
    y = df_engineered[['E']].values
    
    # Data normalization
    print("Normalizing data...")
    scaler_x = StandardScaler()
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    # Save scalers
    joblib.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, "scalers.joblib")
    
    # Split data
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), torch.FloatTensor(y)
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Build model
    print("\n" + "=" * 60)
    print("MODEL CREATION")
    print("=" * 60)
    
    if args.model_type == "standard":
        model = YieldNet(
            input_dim=X.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        print("Created: YieldNet (Standard Residual Network)")
    elif args.model_type == "attention":
        model = YieldNetWithAttention(
            input_dim=X.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.attention_heads,
            dropout=args.dropout
        )
        print("Created: YieldNetWithAttention (Transformer-based)")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        device=str(device),
    )
    
    trainer = Trainer(model, training_config)
    
    try:
        history = trainer.train(train_loader, val_loader)
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model_save_path = os.path.join(
        args.checkpoint_dir,
        f"yield_model_{args.model_type}.pt"
    )
    
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "model_type": args.model_type,
        "input_dim": X.shape[1],
        "input_features": feature_engineer.feature_engineer.feature_cols if hasattr(feature_engineer, 'feature_engineer') else list(df_engineered.drop('E', axis=1).columns)
    }
    
    torch.save(checkpoint, model_save_path)
    print(f"Model saved: {model_save_path}")
    
    # Copy scalers to checkpoint directory
    scalers_save_path = model_save_path.replace(".pt", "_scalers.joblib")
    joblib.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, scalers_save_path)
    print(f"Scalers saved: {scalers_save_path}")
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    trainer.save_history(history_path)
    print(f"Training history saved: {history_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train yield prediction model"
    )
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default=None,
                       help="Path to training CSV file")
    parser.add_argument("--num_samples", type=int, default=8000,
                       help="Number of synthetic samples to generate")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="attention",
                       choices=["standard", "attention"],
                       help="Model architecture type")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of layers")
    parser.add_argument("--attention_heads", type=int, default=8,
                       help="Number of attention heads (for attention model)")
    parser.add_argument("--dropout", type=float, default=0.15,
                       help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3000,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay (L2 regularization)")
    parser.add_argument("--patience", type=int, default=100,
                       help="Early stopping patience")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to train on")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float64"],
                       help="Data type")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory for saving checkpoints")
    
    args = parser.parse_args()
    
    train_model(args)


if __name__ == "__main__":
    main()
