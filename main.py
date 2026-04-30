import os
import re
import torch
import torch.nn
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import customtkinter as ctk
from tkinter import messagebox
import threading

BIO_HSP = np.array([15.2, 4.2, 8.5])
DES_BASE_HSP = np.array([16.5, 15.0, 38.2])
WATER_HSP = np.array([15.5, 16.0, 42.3])
R0_GLY = 12.1
FILE_MODEL, FILE_SX, FILE_SY = 'model.pth', 'sx.joblib', 'sy.joblib'

DFT_DH = -600.76691
DFT_DS = -0.129758
DFT_E_INT = -60.5956
DFT_VOL = 61.067372 
R_GAS = 0.008314 
LOG_FOLDER = 'logs'
LOG_GLYCEROL = 'GLYCEROL.LOG'
LOG_CHOLINE = 'CHOLINE+.LOG'
LOG_DES = {
    1.0: '1-1.LOG',
    2.0: '1-2.LOG',
    3.0: '1-3.LOG'
}


def parse_gaussian_log(path):
    result = {'scf': None, 'enthalpy': None, 'free_energy': None}
    if not os.path.exists(path):
        return result
    with open(path, 'r', encoding='utf-8', errors='ignore') as fd:
        text = fd.read()
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


def load_quantum_references():
    data = {}
    data[LOG_GLYCEROL] = parse_gaussian_log(os.path.join(LOG_FOLDER, LOG_GLYCEROL))
    data[LOG_CHOLINE] = parse_gaussian_log(os.path.join(LOG_FOLDER, LOG_CHOLINE))
    for ratio, filename in LOG_DES.items():
        data[filename] = parse_gaussian_log(os.path.join(LOG_FOLDER, filename))
    glycerol_ref = data.get(LOG_GLYCEROL, {})
    choline_ref = data.get(LOG_CHOLINE, {})
    for filename in LOG_DES.values():
        entry = data.get(filename, {})
        if entry is None:
            continue
        if (glycerol_ref.get('free_energy') is not None and 
            choline_ref.get('free_energy') is not None and 
            entry.get('free_energy') is not None):
            entry['formation_free'] = entry['free_energy'] - (glycerol_ref['free_energy'] + choline_ref['free_energy'])
        else:
            entry['formation_free'] = None
        if (glycerol_ref.get('enthalpy') is not None and 
            choline_ref.get('enthalpy') is not None and 
            entry.get('enthalpy') is not None):
            entry['formation_enthalpy'] = entry['enthalpy'] - (glycerol_ref['enthalpy'] + choline_ref['enthalpy'])
        else:
            entry['formation_enthalpy'] = None
    return data

QUANTUM_DATA = load_quantum_references()

class ResBlock(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.b = torch.nn.Sequential(
            torch.nn.Linear(n, n), 
            torch.nn.LayerNorm(n), 
            torch.nn.Mish(), 
            torch.nn.Dropout(0.15)
        )
    def forward(self, x): return x + self.b(x)

class BioNet(torch.nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.top = torch.nn.Sequential(torch.nn.Linear(in_size, 128), torch.nn.Mish())
        self.res = torch.nn.Sequential(*[ResBlock(128) for _ in range(4)])
        self.out = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x): return self.out(self.res(self.top(x)))

def get_quantum_features(ratio):
    ratios = np.array(sorted(LOG_DES.keys()), dtype=float)
    def interp(field):
        valid_ratios = []
        valid_values = []
        for r in ratios:
            entry = QUANTUM_DATA.get(LOG_DES[r], {})
            val = entry.get(field)
            if val is not None:
                valid_ratios.append(r)
                valid_values.append(val)
        if not valid_values:
            return 0.0
        valid_ratios = np.array(valid_ratios)
        valid_values = np.array(valid_values)
        return float(np.interp(ratio, valid_ratios, valid_values, left=valid_values[0], right=valid_values[-1]))
    return {
        'des_scf': interp('scf'),
        'des_enthalpy': interp('enthalpy'),
        'des_free_energy': interp('free_energy'),
        'formation_free': interp('formation_free'),
        'formation_enthalpy': interp('formation_enthalpy')
    }


def engineer(df):
    df = df.copy()
    df['Inv_T'] = 1000.0 / df['T']
    df['Log_V'] = np.log(df['V'] + 1e-5) 
    df['Kinetic_Barrier'] = df['Log_V'] * df['Inv_T']
    df['Saturation_Index'] = df['G'] / (df['M'] + 1e-4)
    df['Thermo_DG_T'] = DFT_DH - (df['T'] * DFT_DS)
    df['Equilibrium_Proxy'] = np.exp(-(df['Thermo_DG_T'] / 100) / (R_GAS * df['T']))
    df['Adj_Interaction_E'] = DFT_E_INT / (df['R'] + 1e-4)
    df['DES_Volume'] = DFT_VOL
    
    ra_list, red_list = [], []
    des_scf, des_h, des_g = [], [], []
    des_formation_g, des_formation_h = [], []
    for _, row in df.iterrows():
        w_frac = row['W'] / 100.0
        r_mod = 1.0 + (row['R'] - 2.0) * 0.05
        
        eff_hsp = (DES_BASE_HSP * (1 - w_frac) + WATER_HSP * w_frac) * r_mod
        
        dist_sq = 4*(eff_hsp[0] - BIO_HSP[0])**2 + (eff_hsp[1] - BIO_HSP[1])**2 + (eff_hsp[2] - BIO_HSP[2])**2
        ra = np.sqrt(dist_sq) * (298.15 / row['T'])
        
        ra_list.append(ra)
        red_list.append(ra / R0_GLY)

        q = get_quantum_features(row['R'])
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
    df['Quantum_Stability_Index'] = -df['DFT_Formation_DeltaG']
    return df

def train():
    raw = {
        'T': [298.15, 303.15, 313.15, 333.15, 298.15, 303.15, 333.15, 298.15, 313.15],
        'R': [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 2.0, 1.0, 2.0],
        'D': [1.18, 1.17, 1.16, 1.14, 1.18, 1.17, 1.14, 1.25, 1.16],
        'V': [259, 231, 102, 41, 376, 324, 40, 1200, 1500],
        'M': [0.1, 0.15, 0.1, 0.1, 0.2, 0.1, 0.1, 0.05, 0.1],
        'W': [0.05, 0.05, 0.08, 0.1, 0.05, 0.1, 3.5, 0.05, 5.0],
        'G': [0.8, 0.6, 0.5, 0.44, 0.75, 0.5, 0.6, 0.55, 0.55],
        'E': [99.1, 96.5, 92.1, 88.5, 96.6, 97.2, 45.0, 20.0, 15.0]
    }
    df = pd.DataFrame(raw)
    
    aug = []
    for _ in range(8000):
        s = df.sample(n=1).copy().values[0]
        s2 = df.sample(n=1).values[0]
        alpha = np.random.uniform(0, 1)
        synthetic_row = s * alpha + s2 * (1 - alpha)
        
        synthetic_row[0] += np.random.uniform(-2, 2) 
        synthetic_row[3] *= np.random.uniform(0.9, 1.1) 
        
        if synthetic_row[5] > 3.0: synthetic_row[7] *= 0.4
        if synthetic_row[3] > 800: synthetic_row[7] *= 0.6
        
        saturation_index = synthetic_row[6] / (synthetic_row[4] + 1e-4)
        if saturation_index > 12.0:
            synthetic_row[7] *= (12.0 / saturation_index)
            
        synthetic_row[7] = np.clip(synthetic_row[7], 0, 100)
        aug.append(synthetic_row)
    
    f_df = engineer(pd.DataFrame(aug, columns=df.columns))
    x, y = f_df.drop('E', axis=1).values, f_df['E'].values.reshape(-1, 1)
    
    sx, sy = StandardScaler().fit(x), MinMaxScaler(feature_range=(0, 1)).fit(y)
    joblib.dump(sx, FILE_SX); joblib.dump(sy, FILE_SY)

    net = BioNet(x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    xt, yt = torch.FloatTensor(sx.transform(x)), torch.FloatTensor(sy.transform(y))
    
    loss_fn = torch.nn.HuberLoss()
    for epoch in range(3000):
        net.train(); opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward(); opt.step()
        
    torch.save(net.state_dict(), FILE_MODEL)

def predict(t, r, d, v, m, w, g):
    if not os.path.exists(FILE_MODEL): train()

    if d > 2.0 or d < 0.6: raise ValueError(f"Wrong density: {d} g/cm3")
    if t < 273 or t > 500: raise ValueError(f"Temperature {t}K is out of range.")

    row_raw = pd.DataFrame([[t, r, d, v, m, w, g]], columns=['T','R','D','V','M','W','G'])
    f_vec = engineer(row_raw).values
    
    sx, sy = joblib.load(FILE_SX), joblib.load(FILE_SY)
    net = BioNet(f_vec.shape[1])
    try:
        net.load_state_dict(torch.load(FILE_MODEL))
    except Exception:
        train()
        net = BioNet(f_vec.shape[1])
        net.load_state_dict(torch.load(FILE_MODEL))
    
    net.train()
    with torch.no_grad():
        preds = [sy.inverse_transform(net(torch.FloatTensor(sx.transform(f_vec))).numpy())[0][0] for _ in range(100)]
    
    y_avg = np.mean(preds)
    y_std = np.std(preds)
    
    res_gly = g * (1 - (y_avg / 100))
    purity = 100.0 - res_gly
    
    return y_avg, y_std, res_gly, purity

if __name__ == "__main__":
    ctk.set_appearance_mode("light")  # White theme
    ctk.set_default_color_theme("blue")  # Modern theme

    def validate_float(value, name, min_value=None, max_value=None):
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"{name} must be a valid number.")
        if min_value is not None and number < min_value:
            raise ValueError(f"{name} must be at least {min_value}.")
        if max_value is not None and number > max_value:
            raise ValueError(f"{name} must be at most {max_value}.")
        return number

    def calculate_threaded():
        try:
            t = validate_float(entries['Temperature (K)'].get(), 'Temperature (K)', 273, 500)
            r = validate_float(entries['Molar Ratio'].get(), 'Molar Ratio', 0)
            d = validate_float(entries['Density (g/cm3)'].get(), 'Density (g/cm3)', 0.1, 5.0)
            v = validate_float(entries['Viscosity (mPa s)'].get(), 'Viscosity (mPa s)', 0)
            m = validate_float(entries['DES/Oil Mass Ratio'].get(), 'DES/Oil Mass Ratio', 0)
            w = validate_float(entries['Water (%)'].get(), 'Water (%)', 0, 100)
            g = validate_float(entries['Initial Glycerol (%)'].get(), 'Initial Glycerol (%)', 0, 100)

            status_label.configure(text="Calculating...", text_color="blue")
            calc_button.configure(state="disabled")
            root.update_idletasks()

            # Run prediction in thread
            def run_calc():
                try:
                    y_avg, y_std, gly, pur = predict(t, r, d, v, m, w, g)
                    result_text = (
                        f"Yield: {y_avg:.2f}% (±{2 * y_std:.2f}%)\n"
                        f"Residual Glycerol: {gly:.4f}%\n"
                        f"Purity: {pur:.4f}%"
                    )
                    def update_ui():
                        result_label.configure(text=result_text, text_color="black")
                        status_label.configure(
                            text=(
                                "Calculation completed.\n"
                                f"Yield: {y_avg:.2f}%\n"
                                f"Residual Glycerol: {gly:.4f}%\n"
                                f"Purity: {pur:.4f}%"
                            ),
                            text_color="green"
                        )
                        calc_button.configure(state="normal")
                    root.after(0, update_ui)
                except Exception as exc:
                    def update_error():
                        status_label.configure(text="Error occurred.", text_color="red")
                        calc_button.configure(state="normal")
                        messagebox.showerror("Input Error", str(exc))
                    root.after(0, update_error)

            threading.Thread(target=run_calc, daemon=True).start()

        except Exception as exc:
            status_label.configure(text="Error occurred.", text_color="red")
            messagebox.showerror("Input Error", str(exc))
            calc_button.configure(state="normal")

    root = ctk.CTk()
    root.title("Yield Calculator")
    root.geometry("550x700")
    root.resizable(False, False)

    # Main frame with gradient-like background (using CTkFrame)
    main_frame = ctk.CTkFrame(root, fg_color=["#f0f0f0", "#ffffff"], corner_radius=20)
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)

    title_label = ctk.CTkLabel(main_frame, text="Yield Calculator", font=ctk.CTkFont(size=24, weight="bold"))
    title_label.pack(pady=(20, 10))

    # Input frame
    input_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    input_frame.pack(pady=10, padx=20, fill="x")

    labels = [
        "Temperature (K)",
        "Molar Ratio",
        "Density (g/cm3)",
        "Viscosity (mPa s)",
        "DES/Oil Mass Ratio",
        "Water (%)",
        "Initial Glycerol (%)"
    ]

    defaults = [298.15, 2.0, 1.18, 259, 0.1, 0.05, 0.8]
    entries = {}

    for label_text, default_value in zip(labels, defaults):
        row_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        row_frame.pack(fill="x", pady=5)

        lbl = ctk.CTkLabel(row_frame, text=label_text + ":", font=ctk.CTkFont(size=12))
        lbl.pack(side="left", padx=(0, 10))

        entry = ctk.CTkEntry(row_frame, width=200, placeholder_text=str(default_value))
        entry.insert(0, str(default_value))
        entry.pack(side="right")
        entries[label_text] = entry

    # Calculate button with gradient
    calc_button = ctk.CTkButton(
        main_frame,
        text="Calculate",
        command=calculate_threaded,
        fg_color=["#3b8ed0", "#1f6aa5"],  # Gradient blue
        hover_color="#2c7cd1",
        font=ctk.CTkFont(size=14, weight="bold"),
        height=40,
        corner_radius=10
    )
    calc_button.pack(pady=(20, 10))

    # Status label
    status_label = ctk.CTkLabel(
        main_frame,
        text="Enter inputs and click Calculate.",
        font=ctk.CTkFont(size=12),
        wraplength=400,
        justify="left"
    )
    status_label.pack(pady=(0, 10))

    # Result label
    result_label = ctk.CTkLabel(
        main_frame,
        text="Result will appear here after calculation.",
        font=ctk.CTkFont(size=12),
        text_color="black",
        wraplength=400,
        justify="left",
        fg_color="#d0ebff",
        corner_radius=10,
        padx=10,
        pady=10
    )
    result_label.pack(pady=(0, 20), fill="x", padx=20)

    root.mainloop()