"""Utility functions for training and model management"""

import torch
import numpy as np
import random
from typing import Optional


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get torch device"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def get_dtype(dtype: str = "float32") -> torch.dtype:
    """Get torch dtype"""
    if dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Args:
            val_loss: Validation loss
        
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def save_scalers(scaler_x, scaler_y, path: str):
    """Save scalers to file"""
    import joblib
    joblib.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, path)


def load_scalers(path: str):
    """Load scalers from file"""
    import joblib
    data = joblib.load(path)
    return data["scaler_x"], data["scaler_y"]
