#!/usr/bin/env python3
"""Retrain script to build an ensemble of models using the project's training pipeline.

This script reuses the existing `train_model` function from `train.py` and runs it
multiple times with different random seeds to create an ensemble. Each member is
saved in its own checkpoint directory (checkpoints/ensemble_{i}).

Usage examples:
  python retrain.py --train_file training_data.csv --ensemble_size 3 --model_type attention

"""
import argparse
import os
from types import SimpleNamespace

from train import train_model


def main():
    parser = argparse.ArgumentParser(description="Retrain ensemble of yield models")
    parser.add_argument('--train_file', type=str, default='data/training_data.csv', help='CSV file with training data')
    parser.add_argument('--ensemble_size', type=int, default=3, help='Number of ensemble members to train')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Base checkpoint directory')
    parser.add_argument('--model_type', type=str, default='attention', choices=['standard','attention'])
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda'])
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32','float64'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--num_samples', type=int, default=8000)

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for i in range(args.ensemble_size):
        member_seed = args.seed + i
        member_dir = os.path.join(args.checkpoint_dir, f'ensemble_{i+1}')
        os.makedirs(member_dir, exist_ok=True)

        # Build a simple Namespace object expected by train_model
        ns = SimpleNamespace()
        # Copy over values
        for k, v in vars(args).items():
            setattr(ns, k, v)

        # Override per-member settings
        ns.seed = member_seed
        ns.checkpoint_dir = member_dir

        print('\n' + '='*60)
        print(f'Training ensemble member {i+1}/{args.ensemble_size} with seed={member_seed}')
        print(f'Checkpoints will be saved to: {member_dir}')
        print('='*60 + '\n')

        train_model(ns)

    print('\nEnsemble training completed. Models saved under:')
    print(os.path.abspath(args.checkpoint_dir))


if __name__ == '__main__':
    main()
