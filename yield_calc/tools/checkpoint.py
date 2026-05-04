"""Checkpoint management for model training"""

import os
import torch
import json
from typing import Optional, Dict, Any


class CheckpointHandler:
    """Handle model checkpointing and restoration"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        path: Optional[str] = None
    ):
        """Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Training metrics
            path: Custom path (optional)
        """
        if path is None:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        path: str = None
    ) -> Dict[str, Any]:
        """Load model checkpoint
        
        Args:
            model: Model to load into
            optimizer: Optimizer to restore (optional)
            path: Checkpoint path
        
        Returns:
            Checkpoint data
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state"])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        print(f"Checkpoint loaded: {path}")
        return checkpoint
    
    def save_best_model(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, float],
        name: str = "best_model"
    ):
        """Save best model"""
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "metrics": metrics
        }, path)
        print(f"Best model saved: {path}")
    
    def load_best_model(
        self,
        model: torch.nn.Module,
        name: str = "best_model"
    ) -> Dict[str, Any]:
        """Load best model"""
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Best model not found: {path}")
        
        data = torch.load(path, map_location="cpu")
        model.load_state_dict(data["model_state"])
        print(f"Best model loaded: {path}")
        return data
