import streamlit as st
import pandas as pd
import pickle

# Set page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Predictor")

# Load model and data
@st.cache_resource
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

@st.cache_data
def load_data():
    return pd.read_csv('loan_data.csv')

model, label_encoders, feature_names, metrics = load_model()
train_df = load_data()

# Get column info
numeric_cols = [col for col in train_df.columns 
                if col not in ['HOUSE-PRICES', 'Id'] 
                and train_df[col].dtype in ['int64', 'float64']]
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

# Simple form
with st.form("prediction_form"):
    st.write("**Enter house features:**")
    
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    # Numeric features
    with col1:
        for col in numeric_cols[:len(numeric_cols)//2]:
            min_val = float(train_df[col].min())
            max_val = float(train_df[col].max())
            mean_val = float(train_df[col].mean())
            input_data[col] = st.slider(col, min_val, max_val, mean_val)
    
    with col2:
        for col in numeric_cols[len(numeric_cols)//2:]:
            min_val = float(train_df[col].min())
            max_val = float(train_df[col].max())
            mean_val = float(train_df[col].mean())
            input_data[col] = st.slider(col, min_val, max_val, mean_val)
    
    # Categorical features
    for col in categorical_cols:
        vals = sorted(train_df[col].dropna().unique().tolist())
        input_data[col] = st.selectbox(col, vals)
    
    submitted = st.form_submit_button("🔮 Predict Price")

if submitted:
    try:
        # Prepare data
        X = pd.DataFrame([input_data])
        
        # Encode categoricals
        for col in categorical_cols:
            if col in label_encoders:
                X[col] = label_encoders[col].transform([input_data[col]])
        
        X = X[feature_names]
        
        # Predict
        pred = model.predict(X)[0]
        
        st.success(f"### Predicted Price: **${pred:,.0f}**")
        st.info(f"Model R² Score: {metrics['r2_score']:.4f} | RMSE: ${metrics['rmse']:,.0f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
