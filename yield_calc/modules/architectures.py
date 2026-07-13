"""Advanced model architectures for yield prediction"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .blocks import ResidualBlock, TransformerEncoderBlock, DenseBlock


class YieldNet(nn.Module):
    """Enhanced residual network with batch normalization (MACE-inspired)"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        
        # Output layers
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            # Regression output - no sigmoid to allow unconstrained prediction in scaled space
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.output_proj(x)
        return x


class YieldNetWithAttention(nn.Module):
    """Transformer-based model for yield prediction with attention mechanism"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        super().__init__()
        
        # Input projection for each scalar feature token
        self.input_embedding = nn.Linear(1, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Learnable positional embeddings for cls token + each feature token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim + 1, hidden_dim))
        
        # Transformer encoder blocks
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(
                hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_layers)]
        )
        
        # Output head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            # Regression output - no activation
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize positional tokens and embeddings"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention over feature tokens
        
        Args:
            x: Input tensor [batch, input_dim]
        
        Returns:
            Yield prediction [batch, 1]
        """
        B, F = x.shape
        x = x.unsqueeze(-1)  # [B, F, 1]
        x = self.input_embedding(x)  # [B, F, hidden_dim]
        x = self.input_norm(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, F + 1, hidden_dim]
        x = x + self.pos_embed
        
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x


class EnsembleYieldNet(nn.Module):
    """Ensemble of diverse models for robust predictions"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_models: int = 3,
        dropout: float = 0.15
    ):
        super().__init__()
        
        # Create multiple diverse models
        self.models = nn.ModuleList()
        
        for i in range(num_models):
            if i == 0:
                # Standard residual network
                model = YieldNet(input_dim, hidden_dim, num_layers, dropout)
            elif i == 1:
                # Attention-based model
                model = YieldNetWithAttention(
                    input_dim,
                    hidden_dim,
                    num_layers,
                    num_heads=8,
                    dropout=dropout
                )
            else:
                # Deeper residual variant
                model = YieldNet(
                    input_dim,
                    hidden_dim=hidden_dim + 128,
                    num_layers=num_layers + 2,
                    dropout=dropout + 0.05
                )
            
            self.models.append(model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch, input_dim]
        
        Returns:
            mean: Ensemble mean prediction [batch, 1]
            std: Ensemble uncertainty [batch, 1]
        """
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)  # [num_models, batch, 1]
        
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0) + 1e-6  # Add small epsilon for stability
        
        return mean, std


class HybridYieldNet(nn.Module):
    """Hybrid model combining residual blocks and dense connections"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        growth_rate: int = 64,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Initial dense block
        current_dim = hidden_dim
        self.dense_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            dense = DenseBlock(current_dim, growth_rate, dropout)
            self.dense_blocks.append(dense)
            current_dim += growth_rate
            
            # Add transition (compression)
            if i < num_layers - 1:
                self.dense_blocks.append(nn.Sequential(
                    nn.BatchNorm1d(current_dim),
                    nn.ReLU(),
                    nn.Linear(current_dim, hidden_dim)
                ))
                current_dim = hidden_dim
        
        # Global average pooling and output
        self.output = nn.Sequential(
            nn.BatchNorm1d(current_dim),
            nn.ReLU(),
            nn.Linear(current_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            # Regression output - no sigmoid
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = torch.relu(x)
        
        for block in self.dense_blocks:
            x = block(x)
        
        x = self.output(x)
        return x


class TransferLearningYieldNet(nn.Module):
    """Lightweight transfer-learning style regressor for molecular clusters."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        pretrained_dim: Optional[int] = None,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        pretrained_dim = pretrained_dim or input_dim

        self.pretrained_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pretrained_features = self.pretrained_projection(x)
        fused = torch.cat([x, pretrained_features], dim=-1)
        return self.task_head(fused)
