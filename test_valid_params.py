#!/usr/bin/env python3
"""
Quick test script to verify the yield calculator works with valid parameters.
"""

import requests
import json
import time

def test_valid_parameters():
    """Test with parameters within training ranges."""

    # Valid parameters within training ranges
    test_data = {
        "t": 310,      # Temperature: 296-335 K
        "r": 2.0,      # Molar Ratio: 1.0-4.0
        "d": 1.18,     # Density: 1.14-1.25 g/cm³
        "v": 300,      # Viscosity: 36-1645 mPa·s
        "m": 0.1,      # DES/Oil Ratio: 0.05-0.20
        "w": 1.0,      # Water: 0.05-5.0%
        "g": 0.6       # Glycerol: 0.44-0.80%
    }

    print("🧪 Testing Yield Calculator with Valid Parameters")
    print("=" * 50)
    print(f"Input Parameters: {json.dumps(test_data, indent=2)}")
    print()

    try:
        # Start Flask server
        print("🚀 Starting Flask server...")
        import subprocess
        import os

        server_process = subprocess.Popen(
            ["python", "app.py"],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to start
        time.sleep(3)

        # Test health endpoint
        print("🔍 Testing health endpoint...")
        health_response = requests.get("http://localhost:5000/api/health", timeout=5)
        print(f"Health Status: {health_response.status_code}")
        print(f"Health Response: {health_response.json()}")

        # Test prediction endpoint
        print("\n📊 Testing prediction endpoint...")
        predict_response = requests.post(
            "http://localhost:5000/api/predict",
            json=test_data,
            timeout=10
        )

        print(f"Prediction Status: {predict_response.status_code}")

        if predict_response.status_code == 200:
            result = predict_response.json()
            print("✅ Prediction successful!")
            print(f"Yield: {result['yield']:.2f}%")
            print(f"95% CI: ±{result['yield_ci_95']:.2f}%")
            print(f"Residual Glycerol: {result['residual_glycerol']:.4f}%")
            print(f"Purity: {result['purity']:.4f}%")

            # Check if yield is reasonable (should be > 50% with valid params)
            if result['yield'] > 50:
                print("✅ Yield looks reasonable for valid parameters!")
            else:
                print("⚠️ Yield seems low even with valid parameters")

        else:
            print(f"❌ Prediction failed: {predict_response.text}")

    except Exception as e:
        print(f"❌ Test failed: {e}")

    finally:
        # Clean up server
        if 'server_process' in locals():
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    test_valid_parameters()