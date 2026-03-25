import subprocess
import sys
import os
import torch
import torch.nn
import torch.optim
import numpy
import pandas
import joblib
from sklearn.preprocessing import StandardScaler

def setup():
    pkgs = ['torch', 'numpy', 'pandas', 'scikit-learn', 'joblib']
    for p in pkgs:
        try: __import__(p)
        except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", p])

setup()

BIO_HSP = [15.2, 4.2, 8.5]
DES_HSP = [16.5, 15.0, 38.2]
R0_GLY = 12.1
FILE_MODEL, FILE_SX, FILE_SY = 'model.pth', 'sx.joblib', 'sy.joblib'

class ResBlock(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.b = torch.nn.Sequential(
            torch.nn.Linear(n, n), 
            torch.nn.LayerNorm(n), 
            torch.nn.Mish(), 
            torch.nn.Dropout(0.1)
        )
    def forward(self, x): return x + self.b(x)

class BioNet(torch.nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.top = torch.nn.Sequential(torch.nn.Linear(in_size, 128), torch.nn.Mish())
        self.res = torch.nn.Sequential(*[ResBlock(128) for _ in range(3)])
        self.out = torch.nn.Linear(128, 1)
    def forward(self, x): return self.out(self.res(self.top(x)))

def hsp_logic(t):
    dist_sq = 4*(DES_HSP[0] - BIO_HSP[0])**2 + (DES_HSP[1] - BIO_HSP[1])**2 + (DES_HSP[2] - BIO_HSP[2])**2
    ra = numpy.sqrt(dist_sq) * (298.15 / t)
    red = ra / R0_GLY 
    return ra, red

def engineer(df):
    df = df.copy()
    df['Inv_T'] = 1000.0 / df['T']
    df['Log_V'] = numpy.log(df['V'] + 1e-5) 
    df['Kinetic'] = df['Log_V'] * df['Inv_T'] 
    h_res = [hsp_logic(temp) for temp in df['T']]
    df['Ra'], df['RED'] = zip(*h_res)
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
    df = pandas.DataFrame(raw)
    
    aug = []
    for _ in range(6000):
        s = df.sample(n=1).copy()
        s['T'] += numpy.random.normal(0, 0.5)
        s['V'] *= numpy.random.uniform(0.98, 1.02)
        s['E'] = numpy.clip(s['E'] + numpy.random.normal(0, 0.1), 0, 100)
        aug.append(s.values[0])
    
    f_df = engineer(pandas.DataFrame(aug, columns=df.columns))
    x, y = f_df.drop('E', axis=1).values, f_df['E'].values.reshape(-1, 1)
    
    sx, sy = StandardScaler().fit(x), StandardScaler().fit(y)
    joblib.dump(sx, FILE_SX); joblib.dump(sy, FILE_SY)

    net = BioNet(x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=0.0008, weight_decay=1e-4)
    xt, yt = torch.FloatTensor(sx.transform(x)), torch.FloatTensor(sy.transform(y))
    
    for epoch in range(2500):
        net.train(); opt.zero_grad()
        loss = torch.nn.HuberLoss()(net(xt), yt)
        loss.backward(); opt.step()
        
    torch.save(net.state_dict(), FILE_MODEL)

def predict(t, r, d, v, m, w, g):
    if not os.path.exists(FILE_MODEL): train()

    if d > 2.0 or d < 0.6: raise ValueError(f"Physically impossible density: {d} g/cm3")
    if t < 273 or t > 500: raise ValueError(f"Temperature {t}K outside operational range.")

    row_raw = pandas.DataFrame([[t, r, d, v, m, w, g]], columns=['T','R','D','V','M','W','G'])
    f_vec = engineer(row_raw).values
    
    sx, sy = joblib.load(FILE_SX), joblib.load(FILE_SY)
    net = BioNet(f_vec.shape[1])
    net.load_state_dict(torch.load(FILE_MODEL))
    
    net.train() 
    with torch.no_grad():
        preds = [sy.inverse_transform(net(torch.FloatTensor(sx.transform(f_vec))).numpy())[0][0] for _ in range(50)]
    
    y_avg = numpy.clip(numpy.mean(preds), 0, 100)
    y_std = numpy.std(preds)
    
    res_gly = g * (1 - (y_avg / 100))
    purity = 100.0 - res_gly
    
    return y_avg, y_std, res_gly, purity

if __name__ == "__main__":
    try:
        print("\n--- Industrial Biodiesel Purity Analysis ---")
        t = float(input("Temperature (K): "))
        r = float(input("Molar Ratio: "))
        d = float(input("Density (g/cm3): "))
        v = float(input("Viscosity (mPa s): "))
        m = float(input("DES/Oil Mass Ratio: "))
        w = float(input("Water Content (%): "))
        g = float(input("Initial Glycerol (%): "))

        y_avg, y_std, gly, pur = predict(t, r, d, v, m, w, g)
        
        print(f"Extraction Yield:   {y_avg:.2f}% (±{2*y_std:.2f}%)")
        print(f"Residual Glycerol:  {gly:.4f}%")
        print(f"Biodiesel Purity:   {pur:.4f}%")


    except Exception as e: print(f"\n[CRITICAL ERROR]: {e}")