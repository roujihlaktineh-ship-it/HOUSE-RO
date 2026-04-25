from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from functools import lru_cache

app = Flask(__name__)

# Load model and artifacts
@lru_cache(maxsize=1)
def load_model():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('models/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    return model, label_encoders, feature_names, metrics

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'House Price Prediction API',
        'version': '1.0',
        'endpoints': {
            'POST /predict': 'Single house price prediction',
            'POST /predict-batch': 'Batch prediction from CSV',
            'GET /health': 'Health check',
            'GET /metrics': 'Model performance metrics'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    _, _, _, metrics = load_model()
    return jsonify({
        'rmse': float(metrics['rmse']),
        'r2_score': float(metrics['r2_score']),
        'mse': float(metrics['mse'])
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model, label_encoders, feature_names, metrics = load_model()
        
        # Get JSON data
        data = request.get_json()
        
        # Create dataframe
        X = pd.DataFrame([data])
        
        # Identify categorical columns
        categorical_cols = [col for col in X.columns 
                           if col in label_encoders.keys()]
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in label_encoders:
                X[col] = label_encoders[col].transform([data[col]])
        
        # Ensure correct column order
        X = X[feature_names]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'r2_score': float(metrics['r2_score']),
            'rmse': float(metrics['rmse'])
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        model, label_encoders, feature_names, metrics = load_model()
        
        # Get CSV data
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Store IDs if present
        if 'Id' in df.columns:
            ids = df['Id'].tolist()
            X = df.drop(columns=['Id'])
        else:
            ids = list(range(1, len(df) + 1))
            X = df.copy()
        
        # Identify categorical columns
        categorical_cols = [col for col in X.columns 
                           if col in label_encoders.keys()]
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in label_encoders:
                known_classes = set(label_encoders[col].classes_)
                default_class = label_encoders[col].classes_[0]
                X[col] = X[col].fillna(default_class)
                X[col] = X[col].apply(lambda x: x if x in known_classes else default_class)
                X[col] = label_encoders[col].transform(X[col])
        
        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))
        
        # Ensure correct column order
        X = X[feature_names]
        
        # Make predictions
        predictions = model.predict(X)
        
        return jsonify({
            'predictions': [
                {'id': int(ids[i]), 'prediction': float(predictions[i])}
                for i in range(len(predictions))
            ],
            'total': len(predictions)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
