# Yield Calculator

**Advanced ML-based Yield Prediction System**  
*Modular Architecture Inspired by MACE*

## Quick Links

| Link | Purpose |
|------|---------|
| [Render.com](https://render.com) | Deploy backend for free |
| [GitHub](https://github.com/almazorymbetov-nuedukz/yield_calculator) | Push code and collaborate |
| [PyTorch](https://pytorch.org) | Deep learning framework |
| [Flask](https://flask.palletsprojects.com) | Web framework |
| [MACE Paper](https://arxiv.org/abs/2206.07697) | Research inspiration |

## Overview

Yield Calculator is a complete restructuring of the original yield prediction system using an architecture inspired by **MACE** (Multi-Atomic Cluster Expansion). The project implements advanced machine learning algorithms with a modular, scalable design pattern.

## Key Improvements

### 1. **Modular Architecture**
- **Before**: Monolithic `main.py` with all logic mixed together
- **After**: Organized module structure mirroring MACE design:
  ```
  yield_calc/
  ├── modules/     # Neural network models & building blocks
  ├── data/        # Data loading, feature engineering, config
  ├── tools/       # Training, evaluation, utilities
  └── calculators/ # Inference interfaces
  ```

### 2. **Advanced Neural Network Models**

#### YieldNet (Standard)
- Residual blocks with skip connections
- LayerNorm and Mish activation
- Inspired by MACE interaction blocks
- Parameters: 256 hidden channels, 4 layers

#### YieldNetWithAttention (Recommended)
- **Transformer-based** architecture
- Multi-head self-attention mechanism
- Position embeddings
- Feed-forward networks with GeLU activation
- **Better feature learning** through attention
- Uncertainty quantification via ensemble predictions

#### EnsembleYieldNet
- Combines 3 diverse model architectures
- Returns mean prediction + uncertainty (std dev)
- Robust predictions with confidence intervals

### 3. **Enhanced Feature Engineering**
- Increased from 7 → 26 engineered features
- **New features**:
  - Thermodynamic features (Thermo_DG_T, Equilibrium_Proxy)
  - Hansen Solubility Parameter calculations
  - Quantum features (DFT energies, formation energies)
  - Interaction distances and indices
  - Stability indices
  - Advanced interaction energies

### 4. **Sophisticated Training Framework**
- **Trainer class** with:
  - Epoch-by-epoch training with validation
  - Early stopping with patience
  - Training history tracking
  - Checkpoint management
- **Metrics**:
  - MAE, RMSE, Relative MAE, Relative RMSE
  - R² coefficient of determination
- **Utilities**:
  - Random seed control for reproducibility
  - Device management (CPU/GPU)
  - Dtype handling (float32/float64)

### 5. **Inference Interfaces**
- **YieldCalculator**: Single model inference with uncertainty
- **EnsembleCalculator**: Multi-model ensemble predictions
- Batch prediction support
- Automatic scaler handling

### 6. **CLI & Automation**
- **train.py**: Complete training script with arguments:
  ```bash
  python train.py --model_type attention --num_epochs 2000 --batch_size 32
  ```
- **demo_train.py**: Compare model performance
- **test_components.py**: Comprehensive test suite
- **main.py**: Updated GUI using new architecture

## File Structure

```
yield_calculator/
├── yield_calc/               # Main package
│   ├── __init__.py
│   ├── modules/
│   │   ├── blocks.py        # ResidualBlock, AttentionBlock, etc.
│   │   └── architectures.py # Model classes
│   ├── data/
│   │   ├── config.py        # Configuration management
│   │   ├── feature_engineer.py
│   │   └── dataset.py       # PyTorch Dataset wrapper
│   ├── tools/
│   │   ├── train.py         # Trainer class
│   │   ├── metrics.py       # Evaluation metrics
│   │   ├── checkpoint.py    # Model checkpointing
│   │   └── utils.py         # Utilities
│   └── calculators/
│       ├── yield_calculator.py
│       └── ensemble_calculator.py
├── main.py                  # GUI (refactored)
├── train.py                 # Training script
├── demo_train.py           # Performance comparison
├── test_components.py      # Unit tests
├── checkpoints/            # Saved models
└── README.md
```

## Algorithms Implemented

### 1. **Equivariant-Inspired Blocks** (MACE Pattern)
- ResidualBlock: `y = x + f(x)` with LayerNorm + Mish
- Skip connections for gradient flow

### 2. **Attention Mechanism**
- Multi-head self-attention (default: 8 heads)
- Queries, Keys, Values projection
- Softmax normalization + dropout
- Position embeddings

### 3. **Feature Engineering**
- Hansen Solubility Parameters (HSP)
- DFT quantum chemistry features
- Interpolation for ratio-dependent properties
- Temperature/density normalized features

### 4. **Training Techniques**
- AdamW optimizer with weight decay
- Early stopping with patience monitoring
- Learning rate management
- Stochastic dropout for uncertainty

## Usage

### Quick Start

```python
from yield_calc.calculators import YieldCalculator

# Load trained model
calc = YieldCalculator("checkpoints/yield_model_attention.pt", model_type="attention")

# Make prediction
result = calc.predict(
    t=298.15,  # Temperature (K)
    r=2.0,     # Molar ratio
    d=1.18,    # Density (g/cm³)
    v=259,     # Viscosity (mPa·s)
    m=0.1,     # DES/Oil mass ratio
    w=0.05,    # Water (%)
    g=0.8      # Initial glycerol (%)
)

print(f"Yield: {result['yield']:.2f}% (±{result['yield_ci_95']:.2f}%)")
```

### Train New Model

```bash
# Attention-based model (recommended)
python train.py --model_type attention --num_epochs 2000 --batch_size 32

# Standard residual model
python train.py --model_type standard --num_epochs 2000
```

### Start Flask Server

```bash
# Install dependencies (if not already done)
pip install -r requirements-app.txt

# Start the backend server
python app.py
```

Server will run on `http://localhost:5000`

### Web Interface

There are two HTML-based interfaces available:

#### Full Calculator Interface (Recommended)
Open `index.html` in your browser:
- Modern, responsive design
- Real-time input validation  
- Parameter guidance with safe ranges
- Automatic backend connection
- Results with uncertainty estimates
- One-click deployment info

#### Setup & Configuration Interface
Open `setup.html` in your browser:
- Configure backend server URL
- Test backend connection
- View health status
- Deployment guides for Render.com, Heroku

### REST API Endpoints

#### Health Check
```bash
GET /api/health
```
Returns model status and available device (CPU/CUDA)

#### Make Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "t": 315.15,    # Temperature (K) [273-500]
  "r": 2.0,       # Molar Ratio [0-10]
  "d": 1.18,      # Density (g/cm³) [0.6-2.0]
  "v": 259,       # Viscosity (mPa·s) [0-2000]
  "m": 0.1,       # DES/Oil Mass Ratio [0-1]
  "w": 0.05,      # Water (%) [0-100]
  "g": 0.8        # Initial Glycerol (%) [0-100]
}
```

Response:
```json
{
  "yield": 75.43,
  "yield_std": 2.15,
  "yield_ci_95": 4.30,
  "residual_glycerol": 0.197,
  "purity": 99.803
}
```

#### Model Information
```bash
GET /api/info
```
Returns model details, feature names, and parameter ranges

### Run Tests

```bash
python test_components.py
```

Validates all components including model forward passes, training loop, checkpointing.

### Verify Installation

```bash
python check_setup.py
```

Verifies dependencies, model files, and provides quick testing URLs.

### Compare Model Architectures

```bash
python demo_train.py
```

Trains both Standard and Attention models for performance comparison.

### Desktop GUI

```bash
python main.py
```

Launches standalone GUI application (requires customtkinter).

## Deployment Guide

### Development Environment

1. **Backend (Terminal 1):**
   ```bash
   python app.py
   ```

2. **Frontend (Any Browser):**
   - Open `file:///path/to/index.html` (full calculator)
   - Or open `file:///path/to/setup.html` (configuration)

Both interfaces automatically detect `http://localhost:5000`

### Production Deployment

#### Option A: Render.com (Recommended, Free Tier)

1. Push code to GitHub
2. Sign up at [https://render.com](https://render.com)
3. Create new **Web Service**:
   - Connect GitHub repository
   - **Build Command:** `pip install -r requirements-app.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Deploy and get your URL: `https://your-service-name.onrender.com`
5. Update frontend to use your deployed URL

#### Option B: Heroku (Paid, $7+/month)

```bash
heroku login
heroku create your-yield-calculator
git push heroku main
heroku open
```

#### Option C: AWS, Google Cloud, Azure
Use `Procfile` as reference and deploy as standard Python Flask app.

### Security Recommendations

Before deploying to production:
- [ ] Remove `requirements-app.txt` from git (use `requirements.txt` only)
- [ ] Set `debug=False` in production
- [ ] Use HTTPS only for production URLs
- [ ] Implement rate limiting on `/api/predict`
- [ ] Add authentication if accessing sensitive data
- [ ] Monitor logs for errors
- [ ] Keep dependencies updated: `pip install --upgrade -r requirements-app.txt`

## Security & Repository Hygiene

### Protected Files
The following files are excluded from version control (see `.gitignore`):
- `*.joblib` - Trained scaler objects
- `*.pt` - Model weights
- `*.csv` - Training data
- `.env` - Environment variables
- `*.spec` - PyInstaller specs
- `build/`, `dist/` - Build artifacts

### Best Practices
1. **Never commit:**
   - API keys, tokens, or secrets
   - Personal training data
   - Model checkpoints (use `.gitignore`)
   - IDE configuration files

2. **Before public deployment:**
   - Review all code for hardcoded secrets
   - Set `FLASK_ENV=production`
   - Use environment variables for configuration
   - Enable HTTPS on production domains

3. **Data Privacy:**
   - Training data should be kept private
   - Model files can be shared but exclude large checkpoints
   - Use `.gitignore` to prevent accidental commits

### Configuration via Environment Variables

```bash
# .env (not tracked by git)
FLASK_ENV=production
FLASK_DEBUG=0
MODEL_PATH=checkpoints/yield_model_attention.pt
PORT=5000
CORS_ORIGINS=https://yourdomain.com
```

Load in your application:
```python
from dotenv import load_dotenv
import os

load_dotenv()
model_path = os.getenv('MODEL_PATH', 'checkpoints/yield_model_attention.pt')
port = int(os.getenv('PORT', 5000))
```

## Performance Metrics

### Model Architecture Comparison
| Metric | Standard | Attention | Improvement |
|--------|----------|-----------|------------|
| Validation MAE | ~0.67 | ~0.63 | +6% |
| Training Parameters | 65K | 68K | +5% |
| Epochs to Converge | ~400 | ~350 | -12% |
| Inference Speed (CPU) | ~2ms | ~3ms | -50% |

### Advanced Features
- **Uncertainty Quantification**: 100 stochastic passes with MC Dropout
- **Confidence Intervals**: 95% CI from ensemble predictions
- **Residual Glycerol Prediction**: Automatic purity calculation
- **Batch Processing**: Handle multiple predictions simultaneously

## Dependencies

Core:
- `torch` ≥ 1.12
- `numpy`
- `pandas`
- `scikit-learn`
- `customtkinter` (GUI)
- `joblib` (Model serialization)

Optional:
- `cuda` (for GPU acceleration)

## MACE Design Patterns Applied

| MACE Concept | Implementation |
|--------------|----------------|
| **Modular blocks** | ResidualBlock, AttentionBlock, FeedForwardBlock |
| **Message passing** | Multi-head attention mechanism |
| **Equivariance ideas** | Layer normalization for numerical stability |
| **Configuration system** | YieldConfig class with validation |
| **Training tools** | Trainer, CheckpointHandler, metrics module |
| **Inference interface** | Calculator classes (similar to MACE calculators) |
| **Data pipeline** | FeatureEngineer, Dataset, DataLoader |

## Future Enhancements

1. **Graph Neural Networks**: Apply GNN concepts for better feature interaction modeling
2. **Distributed Training**: Multi-GPU support via DistributedDataParallel
3. **Foundation Models**: Pre-train on larger datasets and fine-tune
4. **Hyperparameter Optimization**: Bayesian optimization for model architecture
5. **Deployment**: TorchScript export for production systems
6. **Active Learning**: Uncertainty-based sample selection for data collection

## Testing

Comprehensive test suite (`test_components.py`) validates:
- ✓ Module imports and structure
- ✓ Configuration management
- ✓ Feature engineering pipeline
- ✓ Model forward passes
- ✓ Training utilities
- ✓ Metrics computation
- ✓ Checkpoint save/load
- ✓ Data generation
- ✓ End-to-end training

## References

This implementation is inspired by:
- **MACE**: Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields" (NeurIPS 2022)
- **Transformer Architecture**: Vaswani et al., "Attention Is All You Need" (2017)
- **Deep Residual Learning**: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)

## License

MIT License - See LICENSE file for details

## Contact

For issues, feature requests, or contributions, please open an issue in the repository.

## Support & Resources

### Documentation
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Render Deployment Guide](https://render.com/docs)
- [MACE Repository](https://github.com/ACEsuit/mace)

### Deployment Platforms
- **Render.com** (Free tier): https://render.com
- **Heroku** (Paid): https://heroku.com
- **AWS EC2** (Pay-as-you-go): https://aws.amazon.com
- **Google Cloud Run** (Serverless): https://cloud.google.com/run

### Development Tools
- **Python Package Manager**: `pip install -r requirements-app.txt`
- **Virtual Environment**: `python -m venv .venv`
- **Testing**: `python test_components.py`
- **GUI**: `python main.py`