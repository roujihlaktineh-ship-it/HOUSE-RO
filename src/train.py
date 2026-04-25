import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from config import (
    HOUSE_DATA_FILE, MODELS_DIR, MODEL_FILE, LABEL_ENCODERS_FILE,
    FEATURE_NAMES_FILE, METRICS_FILE, TARGET_COLUMN, ID_COLUMN
)

# Load data
print(f"Loading data from {HOUSE_DATA_FILE}...")
df = pd.read_csv(HOUSE_DATA_FILE)

# Separate features and target
X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
y = df[TARGET_COLUMN]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Handle categorical features
print("\nProcessing categorical features...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = X[col].fillna('Unknown')
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Handle missing values in numeric columns
print("Handling missing values...")
X = X.fillna(X.mean(numeric_only=True))

# Split data
print("\nSplitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("\nTraining Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=1
)
model.fit(X_train, y_train)

# Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save model artifacts
print("\nSaving model artifacts...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

with open(LABEL_ENCODERS_FILE, 'wb') as f:
    pickle.dump(label_encoders, f)

with open(FEATURE_NAMES_FILE, 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# Save metrics
metrics = {
    'rmse': rmse,
    'r2_score': r2,
    'mse': mse
}
with open(METRICS_FILE, 'wb') as f:
    pickle.dump(metrics, f)

print("\nModel artifacts saved to models/:")
print("  - model.pkl")
print("  - label_encoders.pkl")
print("  - feature_names.pkl")
print("  - metrics.pkl")
print("\nTraining complete!")
