"""Neural network building blocks inspired by MACE design patterns"""

import torch
import torch.nn as nn
import math


class ResidualBlock(nn.Module):
    """Residual block with skip connections (inspired by MACE)"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Mish(),  # MACE uses Mish activation
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection: out = x + f(x)"""
        return x + self.dropout(self.net(x))


class AttentionBlock(nn.Module):
    """Multi-head self-attention block"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention
        
        Args:
            x: Input tensor [batch, seq_len, dim]
        
        Returns:
            Attention output [batch, seq_len, dim]
        """
        B, N, C = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FeedForwardBlock(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block: Attention + FeedForward with residuals"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        act_layer: nn.Module = nn.GELU
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionBlock(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForwardBlock(dim, mlp_hidden_dim, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block with residuals"""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DenseBlock(nn.Module):
    """Dense connection block (like DenseNet)"""
    
    def __init__(self, in_dim: int, growth_rate: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, growth_rate),
            nn.Dropout(dropout)
        )
        self.growth_rate = growth_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output concatenated with input (dense connection)"""
        out = self.net(x)
        return torch.cat([x, out], dim=1)
