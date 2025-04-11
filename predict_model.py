import numpy as np
import pandas as pd
import joblib

model = joblib.load('bank_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature columns used in training (same as model.py)
FEATURE_COLUMNS = ['age', 'job', 'marital', 'education', 'default', 'balance',
                   'housing', 'loan', 'contact', 'day', 'month', 'duration',
                   'campaign', 'pdays', 'previous', 'poutcome']

def preprocess_input(data_dict):
    # Convert to DataFrame
    df = pd.DataFrame([data_dict])
    
    # One-hot encode categorical columns just like model.py
    df_encoded = pd.get_dummies(df)

    # Make sure all columns from training are present (fill missing ones with 0)
    for col in scaler.feature_names_in_:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ensure same column order as during training
    df_encoded = df_encoded[scaler.feature_names_in_]

    # Scale
    return scaler.transform(df_encoded)

def predict(data_dict):
    processed_data = preprocess_input(data_dict)
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]  # Probability for class "1" (yes)
    return ("yes (1)" if prediction == 1 else "no (0)", probability)

