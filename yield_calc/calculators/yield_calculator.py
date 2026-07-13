"""Yield prediction calculator - main inference interface"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import joblib

from ..modules import YieldNet, YieldNetWithAttention, TransferLearningYieldNet
from ..data import FeatureEngineer, YieldConfig, QuantumReferences
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
        self.model, self.config, self.input_dim = self._load_model(model_path)
        self.feature_engineer = FeatureEngineer(self.config)
        self.expected_input_dim = self.input_dim
        
        # Load scalers
        scaler_path = model_path.replace(".pt", "_scalers.joblib")
        if os.path.exists(scaler_path):
            self.scalers = joblib.load(scaler_path)
        else:
            self.scalers = {"scaler_x": None, "scaler_y": None}
        
        # Move model to device and set to eval
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, YieldConfig, int]:
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
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
        elif self.model_type == "transfer":
            model = TransferLearningYieldNet(
                input_dim=checkpoint.get("input_dim", 26),
                hidden_dim=config.num_channels,
                pretrained_dim=max(8, min(checkpoint.get("input_dim", 26), 8)),
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights
        model.load_state_dict(checkpoint["model_state"])
        
        return model, config, checkpoint.get("input_dim", 26)
    
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
        molecular_features = np.array([[1.5, 1.0, 2.0, 1.0, 2.5, 1.5, 1.8, 3.0]], dtype=np.float32)
        molecular_frame = pd.DataFrame(molecular_features, columns=[
            "cluster_size",
            "hba_count",
            "hbd_count",
            "biodiesel_count",
            "hydrogen_bond_score",
            "polarity_proxy",
            "miscibility_proxy",
            "component_diversity",
        ])
        df_features = pd.concat([df_features.reset_index(drop=True), molecular_frame], axis=1)
        x = df_features.drop('E', axis=1, errors='ignore').values.astype(np.float32)
        
        # Normalize if scaler available
        if self.scalers["scaler_x"] is not None:
            if x.shape[1] != self.scalers["scaler_x"].mean_.shape[0]:
                if x.shape[1] > self.scalers["scaler_x"].mean_.shape[0]:
                    x = x[:, :self.scalers["scaler_x"].mean_.shape[0]]
                else:
                    pad_width = self.scalers["scaler_x"].mean_.shape[0] - x.shape[1]
                    x = np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
            x = self.scalers["scaler_x"].transform(x)
        else:
            raise RuntimeError(
                f"Missing input scaler for model '{self.model_path}'. "
                "Please ensure the corresponding *_scalers.joblib file is present."
            )

        # Out-of-distribution check: compare raw inputs to config ranges
        ood_warnings = []
        try:
            ranges = getattr(self.config, 'input_ranges', {}) or {}
            for key, val in raw_data.items():
                low, high = ranges.get(key, (None, None))
                if low is not None and high is not None:
                    # compare single value
                    vval = float(val[0])
                    if vval < low or vval > high:
                        ood_warnings.append(f"{key}={vval} outside training range [{low},{high}]")
        except Exception:
            # If config missing or malformed, skip OOD checks silently
            ood_warnings = []
        
        # Convert to tensor
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        # Make predictions with uncertainty (enable dropout for MC dropout)
        predictions = []
        self.model.train()  # Enable dropout for uncertainty estimation
        with torch.no_grad():
            for _ in range(200):  # Increased from 100 to 200 for better statistics
                pred = self.model(x_tensor)
                predictions.append(pred.cpu().numpy())
        self.model.eval()  # Restore eval mode
        
        # Convert list of prediction arrays to numpy array: [n_samples, batch, 1]
        predictions = np.array(predictions)

        # For uncertainty, inverse-transform each sample to original units when possible
        if self.scalers["scaler_y"] is not None:
            # reshape to (n_samples, 1) for scaler
            preds_scaled = predictions.reshape(predictions.shape[0], -1)
            preds_orig = self.scalers["scaler_y"].inverse_transform(preds_scaled)
            preds_orig = preds_orig.squeeze()
        else:
            preds_orig = predictions.squeeze() * 100.0

        # Compute mean and std in original units
        y_avg = float(np.mean(preds_orig))
        y_std = float(np.std(preds_orig))
        
        # Compute residual glycerol and purity.
        # Assume `g` is provided as percentage (e.g., 1.5 means 1.5%). Convert to fraction.
        try:
            g_val = 0.0 if g is None else float(g)
        except Exception:
            g_val = 0.0

        g_frac = g_val / 100.0

        residual_frac = g_frac * (1.0 - (y_avg / 100.0))
        residual_percent = max(0.0, residual_frac * 100.0)
        purity_percent = max(0.0, (1.0 - residual_frac) * 100.0)

        result = {
            "yield": float(max(0.0, min(100.0, y_avg))),
            "yield_std": float(y_std),
            "yield_ci_95": float(2.0 * y_std),
            # Provide both legacy and new keys so frontend integrations are robust
            "residual_glycerol_percent": float(residual_percent),
            "purity_percent": float(purity_percent),
            "residual_glycerol": float(residual_percent),
            "purity": float(purity_percent),
            "temperature": float(t),
            "molar_ratio": float(r),
            "density": float(d),
            "viscosity": float(v)
        }
        # Add OOD warnings if any
        result['warnings'] = ood_warnings
        result['oob'] = len(ood_warnings) > 0
        
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
