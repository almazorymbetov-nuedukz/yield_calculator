"""Metrics computation for model evaluation"""

import torch
import numpy as np
from typing import Tuple


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error"""
    return torch.abs(pred - target).mean().item()


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error"""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def compute_rel_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Relative Mean Absolute Error"""
    mae = torch.abs(pred - target).mean()
    return (mae / (torch.abs(target).mean() + 1e-8)).item()


def compute_rel_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Relative Root Mean Squared Error"""
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    return (rmse / (torch.abs(target).mean() + 1e-8)).item()


def compute_r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """R² coefficient of determination"""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return (1 - (ss_res / (ss_tot + 1e-8))).item()


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> dict:
    """Compute all metrics
    
    Args:
        pred: Predictions [batch, 1]
        target: Targets [batch, 1]
    
    Returns:
        Dictionary with all metrics
    """
    pred = pred.detach()
    target = target.detach()
    
    return {
        "mae": compute_mae(pred, target),
        "rmse": compute_rmse(pred, target),
        "rel_mae": compute_rel_mae(pred, target),
        "rel_rmse": compute_rel_rmse(pred, target),
        "r2": compute_r2_score(pred, target)
    }
