"""Feature engineering for yield prediction"""

import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from .config import QuantumReferences, YieldConfig


class FeatureEngineer:
    """Handles feature extraction and engineering"""
    
    def __init__(self, config: YieldConfig, log_folder: str = "logs"):
        self.config = config
        self.quantum_refs = config.quantum_refs
        self.log_folder = log_folder
        self.quantum_data = self._load_quantum_references()
    
    def _parse_gaussian_log(self, path: str) -> Dict[str, Optional[float]]:
        """Parse Gaussian quantum chemistry log files"""
        result = {'scf': None, 'enthalpy': None, 'free_energy': None}
        
        if not os.path.exists(path):
            return result
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as fd:
                text = fd.read()
        except Exception:
            return result
        
        def last_float(pattern):
            matches = re.findall(pattern, text)
            return float(matches[-1]) if matches else None
        
        result['scf'] = last_float(r'SCF Done:\s*E\([^\)]+\)\s*=\s*([\-\d\.]+)')
        result['enthalpy'] = last_float(r'Sum of electronic and thermal Enthalpies=\s*([\-\d\.]+)')
        result['free_energy'] = last_float(r'Sum of electronic and thermal Free Energies=\s*([\-\d\.]+)')
        
        if result['free_energy'] is None:
            result['free_energy'] = result['scf']
        if result['enthalpy'] is None:
            result['enthalpy'] = result['scf']
        
        return result
    
    def _load_quantum_references(self) -> Dict:
        """Load quantum chemistry reference data"""
        qr = self.quantum_refs
        
        log_files = {
            'GLYCEROL': 'GLYCEROL.LOG',
            'CHOLINE': 'CHOLINE+.LOG',
            'DES_1_1': '1-1.LOG',
            'DES_1_2': '1-2.LOG',
            'DES_1_3': '1-3.LOG'
        }
        
        data = {}
        for key, filename in log_files.items():
            path = os.path.join(self.log_folder, filename)
            data[key] = self._parse_gaussian_log(path)
        
        # Compute formation energies
        glycerol_ref = data.get('GLYCEROL', {})
        choline_ref = data.get('CHOLINE', {})
        
        for key in ['DES_1_1', 'DES_1_2', 'DES_1_3']:
            entry = data.get(key, {})
            if entry is None:
                continue
            
            if (glycerol_ref.get('free_energy') is not None and 
                choline_ref.get('free_energy') is not None and 
                entry.get('free_energy') is not None):
                entry['formation_free'] = (entry['free_energy'] - 
                                         (glycerol_ref['free_energy'] + choline_ref['free_energy']))
            else:
                entry['formation_free'] = None
            
            if (glycerol_ref.get('enthalpy') is not None and 
                choline_ref.get('enthalpy') is not None and 
                entry.get('enthalpy') is not None):
                entry['formation_enthalpy'] = (entry['enthalpy'] - 
                                             (glycerol_ref['enthalpy'] + choline_ref['enthalpy']))
            else:
                entry['formation_enthalpy'] = None
        
        return data
    
    def _get_quantum_features(self, ratio: float) -> Dict[str, float]:
        """Interpolate quantum features for given ratio"""
        ratios = np.array([1.0, 2.0, 3.0], dtype=float)
        
        def interp(field):
            valid_ratios = []
            valid_values = []
            des_keys = ['DES_1_1', 'DES_1_2', 'DES_1_3']
            
            for r, key in zip(ratios, des_keys):
                entry = self.quantum_data.get(key, {})
                val = entry.get(field)
                if val is not None:
                    valid_ratios.append(r)
                    valid_values.append(val)
            
            if not valid_values:
                return 0.0
            
            valid_ratios = np.array(valid_ratios)
            valid_values = np.array(valid_values)
            return float(np.interp(ratio, valid_ratios, valid_values, 
                                  left=valid_values[0], right=valid_values[-1]))
        
        return {
            'des_scf': interp('scf'),
            'des_enthalpy': interp('enthalpy'),
            'des_free_energy': interp('free_energy'),
            'formation_free': interp('formation_free'),
            'formation_enthalpy': interp('formation_enthalpy')
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer features from raw parameters"""
        df = df.copy()
        qr = self.quantum_refs
        
        # Basic transformations
        df['Inv_T'] = 1000.0 / df['T']
        df['Log_V'] = np.log(df['V'] + 1e-5)
        df['Kinetic_Barrier'] = df['Log_V'] * df['Inv_T']
        df['Saturation_Index'] = df['G'] / (df['M'] + 1e-4)
        
        # Thermodynamic features
        df['Thermo_DG_T'] = qr.DFT_DH - (df['T'] * qr.DFT_DS)
        df['Equilibrium_Proxy'] = np.exp(-(df['Thermo_DG_T'] / 100) / (qr.R_GAS * df['T']))
        df['Adj_Interaction_E'] = qr.DFT_E_INT / (df['R'] + 1e-4)
        df['DES_Volume'] = qr.DFT_VOL
        
        # Quantum and solubility features
        ra_list, red_list = [], []
        des_scf, des_h, des_g = [], [], []
        des_formation_g, des_formation_h = [], []
        
        for _, row in df.iterrows():
            w_frac = row['W'] / 100.0
            r_mod = 1.0 + (row['R'] - 2.0) * 0.05
            
            # Hansen Solubility Parameter calculation
            eff_hsp = (qr.DES_BASE_HSP * (1 - w_frac) + qr.WATER_HSP * w_frac) * r_mod
            
            # Interaction distance
            dist_sq = (4 * (eff_hsp[0] - qr.BIO_HSP[0])**2 + 
                      (eff_hsp[1] - qr.BIO_HSP[1])**2 + 
                      (eff_hsp[2] - qr.BIO_HSP[2])**2)
            ra = np.sqrt(dist_sq) * (298.15 / row['T'])
            
            ra_list.append(ra)
            red_list.append(ra / qr.R0_GLY)
            
            # Quantum features
            q = self._get_quantum_features(row['R'])
            des_scf.append(q['des_scf'])
            des_h.append(q['des_enthalpy'])
            des_g.append(q['des_free_energy'])
            des_formation_g.append(q['formation_free'])
            des_formation_h.append(q['formation_enthalpy'])
        
        df['Ra_Dynamic'] = ra_list
        df['RED_Dynamic'] = red_list
        df['DFT_DES_SCF'] = des_scf
        df['DFT_DES_Enthalpy'] = des_h
        df['DFT_DES_FreeEnergy'] = des_g
        df['DFT_Formation_DeltaG'] = des_formation_g
        df['DFT_Formation_DeltaH'] = des_formation_h
        df['Quantum_Stability_Index'] = -np.array(des_formation_g)
        
        # Interaction and stability features
        df['Temp_Normalized_Ratio'] = df['R'] / (df['T'] / 298.15)
        df['Viscosity_Saturation'] = df['V'] * df['Saturation_Index']
        df['Energy_Density'] = df['DFT_Formation_DeltaG'] / (df['DES_Volume'] + 1e-6)
        
        return df
