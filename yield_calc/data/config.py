"""Configuration and constants for yield calculator"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class QuantumReferences:
    """Quantum chemistry reference values (DFT computed)"""
    BIO_HSP: np.ndarray = None  # Hansen Solubility Parameters - Bio compound
    DES_BASE_HSP: np.ndarray = None  # DES base HSP
    WATER_HSP: np.ndarray = None  # Water HSP
    R0_GLY: float = 12.1  # Interaction radius - Glycerol
    
    # Thermodynamic reference data
    DFT_DH: float = -600.76691  # ΔH (enthalpy change)
    DFT_DS: float = -0.129758  # ΔS (entropy change)
    DFT_E_INT: float = -60.5956  # Interaction energy
    DFT_VOL: float = 61.067372  # Volume reference
    
    R_GAS: float = 0.008314  # Gas constant (kJ/mol/K)
    
    def __post_init__(self):
        if self.BIO_HSP is None:
            self.BIO_HSP = np.array([15.2, 4.2, 8.5])
        if self.DES_BASE_HSP is None:
            self.DES_BASE_HSP = np.array([16.5, 15.0, 38.2])
        if self.WATER_HSP is None:
            self.WATER_HSP = np.array([15.5, 16.0, 42.3])


@dataclass
class YieldConfig:
    """Configuration for yield prediction model"""
    
    # Input parameters info
    input_features: List[str] = None
    input_ranges: Dict[str, tuple] = None
    
    # Quantum references
    quantum_refs: QuantumReferences = None
    
    # Model configuration
    num_channels: int = 256
    num_layers: int = 4
    attention_heads: int = 8
    dropout: float = 0.15
    hidden_multiplier: int = 4  # For feedforward dimension in attention
    
    # Data normalization
    use_scaling: bool = True
    scale_output: bool = True
    
    # Device
    device: str = "cpu"
    dtype: str = "float32"
    
    def __post_init__(self):
        if self.input_features is None:
            self.input_features = [
                'T', 'R', 'D', 'V', 'M', 'W', 'G'
            ]
        
        if self.input_ranges is None:
            self.input_ranges = {
                'T': (273, 500),
                'R': (0, 10),
                'D': (0.6, 2.0),
                'V': (0, 2000),
                'M': (0, 1.0),
                'W': (0, 100),
                'G': (0, 100)
            }
        
        if self.quantum_refs is None:
            self.quantum_refs = QuantumReferences()
