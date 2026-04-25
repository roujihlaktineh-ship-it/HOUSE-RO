#!/usr/bin/env python
"""
Diagnose Python environment and sklearn installation
"""
import sys
import os

print("=" * 60)
print("PYTHON ENVIRONMENT DIAGNOSIS")
print("=" * 60)
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Current Directory: {os.getcwd()}")

print("\n" + "=" * 60)
print("CHECKING INSTALLED PACKAGES")
print("=" * 60)

packages = ['pandas', 'numpy', 'sklearn', 'scikit-learn']
for pkg in packages:
    try:
        mod = __import__(pkg)
        location = getattr(mod, '__file__', 'unknown')
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {pkg:20s} - {version} - {location}")
    except ImportError as e:
        print(f"✗ {pkg:20s} - NOT INSTALLED - {e}")

print("\n" + "=" * 60)
print("CHECKING SKLEARN SUBMODULES")
print("=" * 60)

try:
    from sklearn import ensemble
    print(f"✓ sklearn.ensemble imported successfully")
    print(f"  Location: {ensemble.__file__}")
    
    # Check if GradientBoostingRegressor is available
    if hasattr(ensemble, 'GradientBoostingRegressor'):
        print(f"✓ GradientBoostingRegressor is available")
    else:
        print(f"✗ GradientBoostingRegressor NOT found in ensemble")
        print(f"  Available: {[x for x in dir(ensemble) if not x.startswith('_')][:10]}...")
except ImportError as e:
    print(f"✗ sklearn.ensemble import failed: {e}")

print("\n" + "=" * 60)
