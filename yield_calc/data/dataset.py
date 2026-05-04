"""Dataset class for yield prediction"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple, Optional
from .feature_engineer import FeatureEngineer
from .config import YieldConfig


class YieldDataset(TorchDataset):
    """PyTorch Dataset for yield prediction"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_engineer: FeatureEngineer,
        config: YieldConfig,
        input_cols: list,
        output_col: str = 'E',
        scaler: Optional[object] = None
    ):
        """
        Args:
            df: DataFrame with raw data
            feature_engineer: FeatureEngineer instance
            config: YieldConfig instance
            input_cols: List of input column names
            output_col: Target column name
            scaler: Optional fitted scaler for normalization
        """
        self.config = config
        self.feature_engineer = feature_engineer
        
        # Engineer features
        self.df = feature_engineer.engineer_features(df.copy())
        
        # Get feature columns
        self.feature_cols = self.df.columns.drop(output_col).tolist()
        
        # Extract data
        X = self.df[self.feature_cols].values.astype(np.float32)
        y = self.df[[output_col]].values.astype(np.float32)
        
        # Normalize if scaler provided
        if scaler is not None:
            X = scaler.transform(X)
        
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        assert self.X.shape[0] == self.y.shape[0], "X and y must have same length"
    
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def get_feature_names(self) -> list:
        """Return feature column names"""
        return self.feature_cols
