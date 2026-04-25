from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import pandas as pd
import io
import os
from typing import List, Optional
from functools import lru_cache

app = FastAPI(
    title="House Price Prediction API",
    description="ML model API for predicting house prices",
    version="1.0.0"
)

# Load model and artifacts
@lru_cache(maxsize=1)
def load_model():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base_path, 'models/model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base_path, 'models/label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)
    with open(os.path.join(base_path, 'models/feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    with open(os.path.join(base_path, 'models/metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
    return model, label_encoders, feature_names, metrics

# Request/Response models
class PredictionRequest(BaseModel):
    class Config:
        extra = "allow"

class PredictionResponse(BaseModel):
    prediction: float
    r2_score: float
    rmse: float

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    total: int

class HealthResponse(BaseModel):
    status: str

class MetricsResponse(BaseModel):
    rmse: float
    r2_score: float
    mse: float

# Routes
@app.get("/")
async def home():
    return {
        "message": "House Price Prediction API",
        "version": "1.0",
        "endpoints": {
            "POST /predict": "Single house price prediction",
            "POST /predict-batch": "Batch prediction from CSV",
            "GET /health": "Health check",
            "GET /metrics": "Model performance metrics"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "healthy"}

@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    _, _, _, model_metrics = load_model()
    return {
        "rmse": float(model_metrics['rmse']),
        "r2_score": float(model_metrics['r2_score']),
        "mse": float(model_metrics['mse'])
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        model, label_encoders, feature_names, model_metrics = load_model()
        
        # Convert request to dict
        data = request.dict()
        
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
        
        return {
            "prediction": float(prediction),
            "r2_score": float(model_metrics['r2_score']),
            "rmse": float(model_metrics['rmse'])
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    try:
        model, label_encoders, feature_names, model_metrics = load_model()
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
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
        
        return {
            "predictions": [
                {"id": int(ids[i]), "prediction": float(predictions[i])}
                for i in range(len(predictions))
            ],
            "total": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
