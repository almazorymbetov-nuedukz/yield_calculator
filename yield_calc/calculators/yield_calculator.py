"""Yield prediction calculator - main inference interface"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import joblib

from ..modules import YieldNet, YieldNetWithAttention
from ..data import FeatureEngineer, YieldConfig
from ..tools import get_device, get_dtype


class YieldCalculator:
    """Calculator for yield prediction (MACE-like interface)"""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "standard",
        device: str = "cpu",
        dtype: str = "float32"
    ):
        """Initialize calculator with trained model
        
        Args:
            model_path: Path to saved model
            model_type: "standard" or "attention"
            device: "cpu" or "cuda"
            dtype: "float32" or "float64"
        """
        self.model_path = model_path
        self.device = get_device(device)
        self.dtype = get_dtype(dtype)
        self.model_type = model_type
        
        # Load model and configuration
        self.model, self.config = self._load_model(model_path)
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Load scalers
        scaler_path = model_path.replace(".pt", "_scalers.joblib")
        if os.path.exists(scaler_path):
            self.scalers = joblib.load(scaler_path)
        else:
            self.scalers = {"scaler_x": None, "scaler_y": None}
        
        # Move model to device and set to eval
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, YieldConfig]:
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Load config
        config = checkpoint.get("config", YieldConfig())
        
        # Reconstruct model
        if self.model_type == "standard":
            model = YieldNet(
                input_dim=checkpoint.get("input_dim", 26),
                hidden_dim=config.num_channels,
                num_layers=config.num_layers,
                dropout=config.dropout
            )
        elif self.model_type == "attention":
            model = YieldNetWithAttention(
                input_dim=checkpoint.get("input_dim", 26),
                hidden_dim=config.num_channels,
                num_layers=config.num_layers,
                num_heads=config.attention_heads,
                dropout=config.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights
        model.load_state_dict(checkpoint["model_state"])
        
        return model, config
    
    @torch.no_grad()
    def predict(
        self,
        t: float,
        r: float,
        d: float,
        v: float,
        m: float,
        w: float,
        g: float,
        return_uncertainty: bool = False
    ) -> Dict[str, float]:
        """Predict yield with uncertainty estimation
        
        Args:
            t: Temperature (K)
            r: Molar ratio
            d: Density (g/cm³)
            v: Viscosity (mPa·s)
            m: DES/Oil mass ratio
            w: Water (%)
            g: Initial glycerol (%)
            return_uncertainty: Whether to return uncertainty
        
        Returns:
            Dictionary with predictions
        """
        # Validate inputs
        self._validate_inputs(t, d)
        
        # Create input row
        raw_data = {
            'T': [t], 'R': [r], 'D': [d], 'V': [v],
            'M': [m], 'W': [w], 'G': [g]
        }
        df = pd.DataFrame(raw_data)
        
        # Engineer features
        df_features = self.feature_engineer.engineer_features(df)
        x = df_features.drop('E', axis=1, errors='ignore').values.astype(np.float32)
        
        # Normalize if scaler available
        if self.scalers["scaler_x"] is not None:
            x = self.scalers["scaler_x"].transform(x)
        
        # Convert to tensor
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        # Make predictions with uncertainty
        predictions = []
        for _ in range(100):  # 100 stochastic passes with dropout
            pred = self.model(x_tensor)
            predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions).squeeze()
        
        y_avg = np.mean(predictions)
        y_std = np.std(predictions)
        
        # Inverse transform output if scaler available
        if self.scalers["scaler_y"] is not None:
            y_avg = self.scalers["scaler_y"].inverse_transform([[y_avg]])[0, 0]
            y_std_transformed = self.scalers["scaler_y"].inverse_transform([[y_std]])[0, 0]
            y_std = abs(y_std_transformed - y_avg) if y_avg > 0 else y_std
        
        # Compute residual glycerol and purity
        res_gly = g * (1 - (y_avg / 100))
        purity = 100.0 - res_gly
        
        result = {
            "yield": max(0, min(100, y_avg)),
            "yield_std": y_std,
            "yield_ci_95": 2 * y_std,
            "residual_glycerol": max(0, res_gly),
            "purity": max(0, purity),
            "temperature": t,
            "molar_ratio": r,
            "density": d,
            "viscosity": v
        }
        
        return result
    
    def _validate_inputs(self, t: float, d: float):
        """Validate input parameters"""
        if not (273 <= t <= 500):
            raise ValueError(f"Temperature {t}K out of range [273, 500]")
        if not (0.6 <= d <= 2.0):
            raise ValueError(f"Density {d} g/cm³ out of range [0.6, 2.0]")
    
    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction on DataFrame"""
        results = []
        for idx, row in df.iterrows():
            try:
                result = self.predict(
                    t=row['T'], r=row['R'], d=row['D'],
                    v=row['V'], m=row['M'], w=row['W'], g=row['G']
                )
                results.append(result)
            except Exception as e:
                print(f"Error at row {idx}: {e}")
                continue
        
        return pd.DataFrame(results)
