import pandas as pd
import numpy as np
import pickle
import os

# Load model artifacts
print("Loading model artifacts...")
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('models/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Load test data
print("Loading test data...")
test_df = pd.read_csv('test.csv')
test_ids = test_df['Id'].copy()

# Prepare test features
X_test = test_df.drop(columns=['Id'])

# Apply same transformations as training
print("Processing test data...")

# Handle categorical features using fitted encoders
categorical_cols = X_test.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if col in label_encoders:
        # Handle missing values and unseen categories
        known_classes = set(label_encoders[col].classes_)
        default_class = label_encoders[col].classes_[0]  # Use first class as default
        X_test[col] = X_test[col].fillna(default_class)
        X_test[col] = X_test[col].apply(lambda x: x if x in known_classes else default_class)
        X_test[col] = label_encoders[col].transform(X_test[col])

# Handle missing values
X_test = X_test.fillna(X_test.mean(numeric_only=True))

# Ensure same feature order as training
X_test = X_test[feature_names]

# Make predictions
print("Making predictions...")
predictions = model.predict(X_test)

# Create submission dataframe
submission = pd.DataFrame({
    'Id': test_ids,
    'HOUSE-PRICES': predictions
})

# Save predictions
submission.to_csv('predictions.csv', index=False)
print(f"\nPredictions saved to predictions.csv")
print(f"Total predictions: {len(predictions)}")
print(f"Prediction range: ${predictions.min():.2f} - ${predictions.max():.2f}")

# Display training metrics
print("\nModel Metrics from Training:")
print(f"  RMSE: {metrics['rmse']:.2f}")
print(f"  R² Score: {metrics['r2_score']:.4f}")

# Show sample predictions
print("\nSample Predictions:")
print(submission.head(10))
