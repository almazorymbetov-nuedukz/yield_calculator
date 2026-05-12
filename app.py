"""Flask API for Yield Calculator"""

import os
import torch
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from yield_calc.calculators import YieldCalculator

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Model configuration
MODEL_DIR = "checkpoints"
MODEL_ATTENTION = os.path.join(MODEL_DIR, "yield_model_attention.pt")
FALLBACK_MODEL = os.path.join(MODEL_DIR, "best_model.pt")

# Global calculator instance
calculator = None


def initialize_calculator():
    """Initialize calculator with trained model"""
    global calculator
    
    # Try to load the attention model first
    if os.path.exists(MODEL_ATTENTION):
        model_path = MODEL_ATTENTION
        model_type = "attention"
    elif os.path.exists(FALLBACK_MODEL):
        model_path = FALLBACK_MODEL
        model_type = "attention"  # Assume it's attention model
    else:
        return False
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        calculator = YieldCalculator(model_path, model_type=model_type, device=device)
        return True
    except Exception as e:
        print(f"Error initializing calculator: {e}")
        return False


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': calculator is not None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict yield based on input parameters
    
    Expected JSON body:
    {
        "t": float,  # Temperature (K)
        "r": float,  # Molar Ratio
        "d": float,  # Density (g/cm3)
        "v": float,  # Viscosity (mPa s)
        "m": float,  # DES/Oil Mass Ratio
        "w": float,  # Water (%)
        "g": float   # Initial Glycerol (%)
    }
    """
    if calculator is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        # Validate inputs
        required_fields = ['t', 'r', 'd', 'v', 'm', 'w', 'g']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Extract parameters
        t = float(data['t'])  # Temperature
        r = float(data['r'])  # Molar Ratio
        d = float(data['d'])  # Density
        v = float(data['v'])  # Viscosity
        m = float(data['m'])  # DES/Oil Mass Ratio
        w = float(data['w'])  # Water
        g = float(data['g'])  # Glycerol
        
        # Validate ranges
        if not (273 <= t <= 500):
            return jsonify({'error': 'Temperature must be between 273 and 500 K'}), 400
        if d < 0.1 or d > 5.0:
            return jsonify({'error': 'Density must be between 0.1 and 5.0 g/cm3'}), 400
        if w < 0 or w > 100:
            return jsonify({'error': 'Water content must be between 0 and 100%'}), 400
        if g < 0 or g > 100:
            return jsonify({'error': 'Glycerol content must be between 0 and 100%'}), 400
        
        # Make prediction
        result = calculator.predict(t, r, d, v, m, w, g)
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get information about the calculator"""
    return jsonify({
        'name': 'Yield Calculator',
        'version': '2.0',
        'description': 'Advanced ML-based yield prediction with transformer architecture',
        'features': [
            'Transformer-based attention networks',
            'Uncertainty quantification via ensemble',
            'Advanced feature engineering (26 features)',
            'GPU acceleration support',
            'Real-time predictions'
        ],
        'parameters': {
            'temperature': {'min': 273, 'max': 500, 'unit': 'K', 'description': 'Reaction temperature'},
            'molar_ratio': {'min': 0, 'unit': 'ratio', 'description': 'Molar ratio of reactants'},
            'density': {'min': 0.1, 'max': 5.0, 'unit': 'g/cm3', 'description': 'Reaction mixture density'},
            'viscosity': {'min': 0, 'unit': 'mPa·s', 'description': 'Dynamic viscosity'},
            'des_oil_ratio': {'min': 0, 'unit': 'ratio', 'description': 'DES to oil mass ratio'},
            'water': {'min': 0, 'max': 100, 'unit': '%', 'description': 'Water content'},
            'glycerol': {'min': 0, 'max': 100, 'unit': '%', 'description': 'Initial glycerol content'}
        }
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize calculator on startup
    print("Initializing calculator...")
    if initialize_calculator():
        print("✓ Calculator initialized successfully")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    else:
        print("✗ Warning: Could not initialize calculator")
    
    # Run Flask app
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
