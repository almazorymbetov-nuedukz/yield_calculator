"""Ensemble calculator for robust predictions"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import joblib

from ..modules import YieldNet, YieldNetWithAttention, EnsembleYieldNet
from ..data import FeatureEngineer, YieldConfig
from ..tools import get_device, get_dtype


class EnsembleCalculator:
    """Ensemble calculator using multiple model architectures"""
    
    def __init__(
        self,
        model_paths: List[str],
        device: str = "cpu",
        dtype: str = "float32"
    ):
        """Initialize ensemble calculator
        
        Args:
            model_paths: List of paths to trained models
            device: "cpu" or "cuda"
            dtype: "float32" or "float64"
        """
        self.device = get_device(device)
        self.dtype = get_dtype(dtype)
        
        # Load all models
        self.models = []
        self.configs = []
        
        for model_path in model_paths:
            model, config = self._load_model(model_path)
            self.models.append(model)
            self.configs.append(config)
        
        # Use first config for feature engineering
        self.config = self.configs[0]
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Load scalers from first model
        scaler_path = model_paths[0].replace(".pt", "_scalers.joblib")
        if os.path.exists(scaler_path):
            self.scalers = joblib.load(scaler_path)
        else:
            self.scalers = {"scaler_x": None, "scaler_y": None}
    
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, YieldConfig]:
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location="cpu")
        
        config = checkpoint.get("config", YieldConfig())
        model_type = checkpoint.get("model_type", "standard")
        
        if model_type == "standard":
            model = YieldNet(
                input_dim=checkpoint.get("input_dim", 26),
                hidden_dim=config.num_channels,
                num_layers=config.num_layers,
                dropout=config.dropout
            )
        elif model_type == "attention":
            model = YieldNetWithAttention(
                input_dim=checkpoint.get("input_dim", 26),
                hidden_dim=config.num_channels,
                num_layers=config.num_layers,
                num_heads=config.attention_heads,
                dropout=config.dropout
            )
        else:
            model = YieldNet(
                input_dim=checkpoint.get("input_dim", 26),
                hidden_dim=config.num_channels,
                num_layers=config.num_layers,
                dropout=config.dropout
            )
        
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()
        
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
        g: float
    ) -> Dict[str, float]:
        """Ensemble prediction with uncertainty quantification
        
        Returns averaged predictions and uncertainty from multiple models
        """
        # Validate inputs
        if not (273 <= t <= 500):
            raise ValueError(f"Temperature {t}K out of range")
        if not (0.6 <= d <= 2.0):
            raise ValueError(f"Density {d} g/cm³ out of range")
        
        # Create input row
        raw_data = {
            'T': [t], 'R': [r], 'D': [d], 'V': [v],
            'M': [m], 'W': [w], 'G': [g]
        }
        df = pd.DataFrame(raw_data)
        
        # Engineer features
        df_features = self.feature_engineer.engineer_features(df)
        x = df_features.drop('E', axis=1, errors='ignore').values.astype(np.float32)
        
        # Normalize
        if self.scalers["scaler_x"] is not None:
            x = self.scalers["scaler_x"].transform(x)
        
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        # Get predictions from all models
        all_predictions = []
        
        for model in self.models:
            # Multiple passes for uncertainty
            for _ in range(50):
                pred = model(x_tensor).cpu().numpy()
                all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions).squeeze()
        
        # Compute statistics
        y_mean = np.mean(all_predictions)
        y_std = np.std(all_predictions)
        y_median = np.median(all_predictions)
        y_min = np.percentile(all_predictions, 2.5)
        y_max = np.percentile(all_predictions, 97.5)
        
        # Inverse transform if scaler available
        if self.scalers["scaler_y"] is not None:
            y_mean = self.scalers["scaler_y"].inverse_transform([[y_mean]])[0, 0]
            y_median = self.scalers["scaler_y"].inverse_transform([[y_median]])[0, 0]
            y_min = self.scalers["scaler_y"].inverse_transform([[y_min]])[0, 0]
            y_max = self.scalers["scaler_y"].inverse_transform([[y_max]])[0, 0]
        
        y_mean = max(0, min(100, y_mean))
        y_median = max(0, min(100, y_median))
        
        # Compute residual and purity
        res_gly = g * (1 - (y_mean / 100))
        purity = 100.0 - res_gly
        
        return {
            "yield_mean": y_mean,
            "yield_median": y_median,
            "yield_std": y_std,
            "yield_ci_95_lower": max(0, y_min),
            "yield_ci_95_upper": min(100, y_max),
            "residual_glycerol": max(0, res_gly),
            "purity": max(0, purity),
            "num_models": len(self.models),
            "num_predictions": len(all_predictions)
        }
    
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
        
        return pd.DataFrame(results)
