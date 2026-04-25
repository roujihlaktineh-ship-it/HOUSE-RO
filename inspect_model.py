#!/usr/bin/env python
"""
Inspect and display model artifacts from pickle files
"""
import pickle
import os
from pathlib import Path

# Setup paths
MODELS_DIR = Path(__file__).parent / 'models'

def inspect_model():
    """Display model information"""
    model_file = MODELS_DIR / 'model.pkl'
    if not model_file.exists():
        print(f"❌ Model not found: {model_file}")
        return
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Model Type: {type(model).__name__}")
    print(f"Model: {model}")
    if hasattr(model, 'n_estimators'):
        print(f"  - Estimators: {model.n_estimators}")
    if hasattr(model, 'learning_rate'):
        print(f"  - Learning Rate: {model.learning_rate}")
    if hasattr(model, 'max_depth'):
        print(f"  - Max Depth: {model.max_depth}")

def inspect_features():
    """Display feature names"""
    features_file = MODELS_DIR / 'feature_names.pkl'
    if not features_file.exists():
        print(f"❌ Features not found: {features_file}")
        return
    
    with open(features_file, 'rb') as f:
        features = pickle.load(f)
    
    print("\n" + "="*60)
    print("FEATURE INFORMATION")
    print("="*60)
    print(f"Total Features: {len(features)}")
    print(f"Features:\n{features}")

def inspect_encoders():
    """Display label encoders"""
    encoders_file = MODELS_DIR / 'label_encoders.pkl'
    if not encoders_file.exists():
        print(f"❌ Encoders not found: {encoders_file}")
        return
    
    with open(encoders_file, 'rb') as f:
        encoders = pickle.load(f)
    
    print("\n" + "="*60)
    print("LABEL ENCODERS")
    print("="*60)
    print(f"Total Encoded Features: {len(encoders)}")
    print("\nCategorical Features & Classes:")
    for feature, encoder in sorted(encoders.items()):
        classes = encoder.classes_
        print(f"\n  {feature}:")
        print(f"    Classes: {list(classes)}")
        print(f"    Count: {len(classes)}")

def inspect_metrics():
    """Display model metrics"""
    metrics_file = MODELS_DIR / 'metrics.pkl'
    if not metrics_file.exists():
        print(f"❌ Metrics not found: {metrics_file}")
        return
    
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    
    print("\n" + "="*60)
    print("MODEL METRICS")
    print("="*60)
    for metric_name, metric_value in sorted(metrics.items()):
        if isinstance(metric_value, float):
            print(f"{metric_name.upper():20s}: {metric_value:,.2f}")
        else:
            print(f"{metric_name.upper():20s}: {metric_value}")

def main():
    """Run all inspections"""
    print("\n🔍 INSPECTING MODEL ARTIFACTS\n")
    print(f"Models directory: {MODELS_DIR}\n")
    
    # List available files
    if MODELS_DIR.exists():
        files = list(MODELS_DIR.glob('*.pkl'))
        print(f"Found {len(files)} pickle files:")
        for f in sorted(files):
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.2f} KB)")
    
    # Inspect all artifacts
    inspect_model()
    inspect_features()
    inspect_encoders()
    inspect_metrics()
    
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    main()
