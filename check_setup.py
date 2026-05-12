#!/usr/bin/env python3
"""
Quick Test Script - Verify Yield Calculator Setup
"""

import os
import sys
import json
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    required_files = {
        'Frontend': [
            'index.html',
            'setup.html'
        ],
        'Backend': [
            'app.py',
            'requirements-app.txt',
            'Procfile'
        ],
        'Documentation': [
            'WEBSITE_README.md',
            'GETTING_STARTED.md',
            'README-DEPLOY.md',
            'CONFIG_GUIDE.md',
            'PACKAGE_SUMMARY.md'
        ],
        'Configuration': [
            '.gitignore',
            '.github/workflows/deploy.yml',
            '.github/workflows/backend-deploy.yml'
        ]
    }
    
    print("📋 Checking Yield Calculator Setup\n")
    print("=" * 60)
    
    all_good = True
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file in files:
            path = Path(file)
            if path.exists():
                size = path.stat().st_size
                if size > 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                print(f"  ✅ {file:<40} ({size_str})")
            else:
                print(f"  ❌ {file:<40} (MISSING)")
                all_good = False
    
    print("\n" + "=" * 60)
    
    # Check models
    print("\n🤖 ML Models:")
    models = [
        'checkpoints/yield_model_attention.pt',
        'checkpoints/yield_model_attention_scalers.joblib'
    ]
    for model in models:
        path = Path(model)
        if path.exists():
            size = path.stat().st_size / (1024*1024)
            print(f"  ✅ {model:<40} ({size:.1f}MB)")
        else:
            print(f"  ⚠️  {model:<40} (optional)")
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("\n✨ All required files are present!")
        print("\n📖 Next Steps:")
        print("  1. Read: WEBSITE_README.md")
        print("  2. Test: python app.py")
        print("  3. Deploy: Follow README-DEPLOY.md")
        return 0
    else:
        print("\n❌ Some files are missing!")
        return 1

def check_dependencies():
    """Check Python dependencies"""
    print("\n🔍 Checking Dependencies...\n")
    
    required = {
        'Flask': 'flask',
        'Flask-CORS': 'flask_cors',
        'PyTorch': 'torch',
        'NumPy': 'numpy',
        'scikit-learn': 'sklearn',
        'joblib': 'joblib'
    }
    
    missing = []
    for name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (install: pip install {import_name})")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing: {', '.join(missing)}")
        print(f"   Install with: pip install -r requirements-app.txt")
        return False
    else:
        print("\n✨ All dependencies installed!")
        return True

def print_urls():
    """Print helpful URLs"""
    print("\n" + "=" * 60)
    print("\n🌐 Quick URLs:")
    print("  Frontend:  file:///path/to/index.html")
    print("  Setup:     file:///path/to/setup.html")
    print("  Backend:   http://localhost:5000")
    print("  Health:    http://localhost:5000/api/health")

def main():
    """Run all checks"""
    print("\n" + "🎯 " * 15)
    print("\nYIELD CALCULATOR - SETUP CHECK")
    print("\n" + "🎯 " * 15 + "\n")
    
    files_ok = check_files() == 0
    deps_ok = check_dependencies()
    print_urls()
    
    print("\n" + "=" * 60)
    
    if files_ok and deps_ok:
        print("\n✅ Everything looks good! Ready to deploy.")
        print("\n📚 Documentation:")
        print("  • WEBSITE_README.md      - Complete overview")
        print("  • GETTING_STARTED.md     - Quick start")
        print("  • README-DEPLOY.md       - Deployment guide")
        print("  • CONFIG_GUIDE.md        - Configuration")
        print("  • PACKAGE_SUMMARY.md     - This package")
        print("\n🚀 Start with:")
        print("  python app.py")
        return 0
    else:
        print("\n⚠️  Please install missing dependencies:")
        print("  pip install -r requirements-app.txt")
        return 1

if __name__ == '__main__':
    sys.exit(main())
