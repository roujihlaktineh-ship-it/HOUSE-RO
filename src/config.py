"""
Configuration module for House Price Prediction project.
Defines paths for models, data, and other resources.
"""

import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Model paths
MODELS_DIR = PROJECT_ROOT / 'models'
MODEL_FILE = MODELS_DIR / 'model.pkl'
LABEL_ENCODERS_FILE = MODELS_DIR / 'label_encoders.pkl'
FEATURE_NAMES_FILE = MODELS_DIR / 'feature_names.pkl'
METRICS_FILE = MODELS_DIR / 'metrics.pkl'

# Data paths
DATA_DIR = PROJECT_ROOT
HOUSE_DATA_FILE = DATA_DIR / 'HOUSE_data.csv'
TRAIN_FILE = DATA_DIR / 'train.csv'
TEST_FILE = DATA_DIR / 'test.csv'
PREDICTIONS_FILE = DATA_DIR / 'predictions.csv'

# API configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 8000
STREAMLIT_PORT = 8502

# Model configuration
TARGET_COLUMN = 'HOUSE-PRICES'
ID_COLUMN = 'Id'
SALE_PRICE_COLUMN = 'SalePrice'

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Print paths for debugging (optional)
if __name__ == '__main__':
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"House Data File: {HOUSE_DATA_FILE}")
    print(f"Train File: {TRAIN_FILE}")
    print(f"Test File: {TEST_FILE}")
