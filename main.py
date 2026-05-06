"""Yield Calculator GUI - Using Modular Architecture"""

import os
import torch
import customtkinter as ctk
from tkinter import messagebox, filedialog
import threading

# Import from new modular structure
from yield_calc.calculators import YieldCalculator, EnsembleCalculator
from yield_calc.tools import set_random_seed

# Model paths
MODEL_DIR = "checkpoints"
MODEL_STANDARD = os.path.join(MODEL_DIR, "yield_model_standard.pt")
MODEL_ATTENTION = os.path.join(MODEL_DIR, "yield_model_attention.pt")

# Initialize random seed
set_random_seed(42)

# Global calculator instance
calculator = None


def initialize_calculator(model_type: str = "attention"):
    """Initialize calculator with trained model"""
    global calculator
    
    # Prioritize the properly trained model with config
    primary_model = MODEL_ATTENTION if model_type == "attention" else MODEL_STANDARD
    fallback_model = os.path.join(MODEL_DIR, "best_model.pt")
    
    if os.path.exists(primary_model):
        model_path = primary_model
    elif os.path.exists(fallback_model):
        model_path = fallback_model
        print(f"Warning: Using fallback model {fallback_model}. Consider training with: python train.py --model_type {model_type}")
    else:
        raise FileNotFoundError(
            f"No trained model found.\n\n"
            f"Please train a model first by running:\n"
            f"  python train.py --model_type {model_type} --train_file training_data.csv\n"
            f"  python train.py --model_type attention --train_file training_data.csv (recommended)"
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculator = YieldCalculator(model_path, model_type=model_type, device=device)


def train_and_save_model():
    """Train model using the CLI script"""
    try:
        import subprocess
        result = messagebox.showinfo(
            "Training Required",
            "A trained model is required. The training script will now be executed.\n\n"
            "This may take a few minutes depending on your hardware.",
            icon=messagebox.INFO
        )
        
        # Run training script
        subprocess.run([
            "python", "train.py",
            "--model_type", "attention",
            "--num_epochs", "1000",
            "--batch_size", "32"
        ])
        
        initialize_calculator("attention")
        return True
    except Exception as e:
        messagebox.showerror("Training Error", f"Failed to train model:\n{e}")
        return False


def predict(t: float, r: float, d: float, v: float, m: float, w: float, g: float):
    """Make prediction using calculator"""
    global calculator
    
    if calculator is None:
        try:
            initialize_calculator("attention")
        except FileNotFoundError as e:
            messagebox.showerror("Model Not Found", str(e))
            return None
    
    return calculator.predict(t, r, d, v, m, w, g)


if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    
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
            
            def run_calc():
                try:
                    result = predict(t, r, d, v, m, w, g)
                    if result is None:
                        raise ValueError("Prediction failed")
                    
                    result_text = (
                        f"Yield: {result['yield']:.2f}% (±{result['yield_ci_95']:.2f}%)\n"
                        f"Residual Glycerol: {result['residual_glycerol']:.4f}%\n"
                        f"Purity: {result['purity']:.4f}%"
                    )
                    
                    def update_ui():
                        result_label.configure(text=result_text, text_color="black")
                        status_label.configure(
                            text=(
                                "Calculation completed.\n"
                                f"Yield: {result['yield']:.2f}%\n"
                                f"Residual Glycerol: {result['residual_glycerol']:.4f}%\n"
                                f"Purity: {result['purity']:.4f}%"
                            ),
                            text_color="green"
                        )
                        calc_button.configure(state="normal")
                    
                    root.after(0, update_ui)
                except Exception as exc:
                    def update_error(exc=exc):
                        status_label.configure(text="Error occurred.", text_color="red")
                        calc_button.configure(state="normal")
                        messagebox.showerror("Prediction Error", str(exc))
                    
                    root.after(0, update_error)
            
            threading.Thread(target=run_calc, daemon=True).start()
        
        except Exception as exc:
            status_label.configure(text="Error occurred.", text_color="red")
            messagebox.showerror("Input Error", str(exc))
            calc_button.configure(state="normal")
    
    def show_about():
        messagebox.showinfo(
            "About Yield Calculator",
            "Yield Calculator v2.0\n\n"
            "Advanced ML-based yield prediction with modular architecture\n"
            "Inspired by MACE (Multi-Atomic Cluster Expansion)\n\n"
            "Features:\n"
            "• Transformer-based attention networks\n"
            "• Uncertainty quantification\n"
            "• Advanced feature engineering\n"
            "• Modular, scalable architecture"
        )
    
    # Main window
    root = ctk.CTk()
    root.title("Yield Calculator v2.0")
    root.geometry("600x800")
    root.resizable(False, False)
    
    # Main frame
    main_frame = ctk.CTkFrame(root, fg_color=["#f0f0f0", "#ffffff"], corner_radius=20)
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Title
    title_label = ctk.CTkLabel(
        main_frame,
        text="Yield Calculator v2.0",
        font=ctk.CTkFont(size=24, weight="bold")
    )
    title_label.pack(pady=(20, 5))
    
    subtitle_label = ctk.CTkLabel(
        main_frame,
        text="ML-based prediction with modular architecture",
        font=ctk.CTkFont(size=12, slant="italic"),
        text_color="gray"
    )
    subtitle_label.pack(pady=(0, 20))
    
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
        
        lbl = ctk.CTkLabel(row_frame, text=label_text + ":", font=ctk.CTkFont(size=11), width=180)
        lbl.pack(side="left", padx=(0, 10))
        
        entry = ctk.CTkEntry(row_frame, width=150, placeholder_text=str(default_value))
        entry.insert(0, str(default_value))
        entry.pack(side="right")
        entries[label_text] = entry
    
    # Buttons frame
    button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    button_frame.pack(pady=20, padx=20, fill="x")
    
    calc_button = ctk.CTkButton(
        button_frame,
        text="Calculate Yield",
        command=calculate_threaded,
        fg_color=["#3b8ed0", "#1f6aa5"],
        hover_color="#2c7cd1",
        font=ctk.CTkFont(size=13, weight="bold"),
        height=40,
        corner_radius=10
    )
    calc_button.pack(side="left", padx=5, fill="x", expand=True)
    
    about_button = ctk.CTkButton(
        button_frame,
        text="ℹ",
        command=show_about,
        font=ctk.CTkFont(size=14),
        width=40,
        height=40,
        corner_radius=10
    )
    about_button.pack(side="right", padx=5)
    
    # Status label
    status_label = ctk.CTkLabel(
        main_frame,
        text="Ready. Enter parameters and click Calculate.",
        font=ctk.CTkFont(size=11),
        wraplength=400,
        justify="left",
        text_color="gray"
    )
    status_label.pack(pady=(0, 10), padx=20)
    
    # Result label
    result_label = ctk.CTkLabel(
        main_frame,
        text="Results will appear here.",
        font=ctk.CTkFont(size=12),
        text_color="black",
        wraplength=400,
        justify="left",
        fg_color="#d0ebff",
        corner_radius=10,
        padx=15,
        pady=15
    )
    result_label.pack(pady=(0, 20), fill="x", padx=20)
    
    # Info frame
    info_frame = ctk.CTkFrame(main_frame, fg_color="#f5f5f5", corner_radius=10)
    info_frame.pack(pady=10, padx=20, fill="x")
    
    info_label = ctk.CTkLabel(
        info_frame,
        text="Model: Attention-based Neural Network | Device: " + ("GPU" if torch.cuda.is_available() else "CPU"),
        font=ctk.CTkFont(size=10),
        text_color="gray"
    )
    info_label.pack(pady=10, padx=10)
    
    # Initialize calculator on first calculation attempt
    root.mainloop()