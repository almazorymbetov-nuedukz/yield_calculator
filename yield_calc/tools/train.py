"""Training loop and trainer class"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Dict, Tuple, Callable
from dataclasses import dataclass
import json
import os

from .metrics import compute_metrics
from .checkpoint import CheckpointHandler
from .utils import EarlyStopping


@dataclass
class TrainingConfig:
    """Configuration for training"""
    num_epochs: int = 3000
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 100
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"
    num_eval_intervals: int = 10  # Evaluate every N epochs


class Trainer:
    """Trainer class for model training (MACE-inspired)"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[Callable] = None
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        self.loss_fn = loss_fn or nn.MSELoss()
        self.checkpoint_handler = CheckpointHandler(config.checkpoint_dir)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.early_stopping = EarlyStopping(patience=config.patience)
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Train for one epoch
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Average loss and metrics
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * x.shape[0]
            all_preds.append(pred.detach())
            all_targets.append(y.detach())
        
        avg_loss = total_loss / len(train_loader.dataset)
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate on validation set
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average loss and metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            
            total_loss += loss.item() * x.shape[0]
            all_preds.append(pred)
            all_targets.append(y)
        
        avg_loss = total_loss / len(val_loader.dataset)
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: Optional[LRScheduler] = None
    ):
        """Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            scheduler: Learning rate scheduler (optional)
        """
        eval_interval = max(1, self.config.num_epochs // self.config.num_eval_intervals)
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Evaluate periodically
            if (epoch + 1) % eval_interval == 0 or epoch == 0:
                val_loss, val_metrics = self.evaluate(val_loader)
                
                self.training_history["train_loss"].append(train_loss)
                self.training_history["val_loss"].append(val_loss)
                self.training_history["train_metrics"].append(train_metrics)
                self.training_history["val_metrics"].append(val_metrics)
                
                if (epoch + 1) % (eval_interval * 2) == 0:
                    print(f"Epoch {epoch+1}/{self.config.num_epochs}")
                    print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                    print(f"  Train MAE: {train_metrics['mae']:.4f} | Val MAE: {val_metrics['mae']:.4f}")
                    print(f"  Train R²: {train_metrics['r2']:.4f} | Val R²: {val_metrics['r2']:.4f}")
                
                # Save best model
                if len(self.training_history["val_loss"]) == 1 or val_loss < min(self.training_history["val_loss"][:-1]):
                    self.checkpoint_handler.save_best_model(self.model, val_metrics)
                
                # Early stopping
                if self.early_stopping(val_loss):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
        
        print("Training completed!")
        return self.training_history
    
    def save_history(self, path: str):
        """Save training history"""
        with open(path, 'w') as f:
            # Convert tensors to python types for JSON serialization
            history = {}
            for key, val in self.training_history.items():
                if key in ["train_loss", "val_loss"]:
                    history[key] = val
                else:
                    history[key] = [dict(m) for m in val]
            json.dump(history, f, indent=2)
