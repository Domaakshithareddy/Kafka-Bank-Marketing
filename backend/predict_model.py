import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Form
from typing import Annotated

app = FastAPI()

model = joblib.load("backend/bank_model.pkl")
scaler = joblib.load("backend/scaler.pkl")

# Feature columns used in training
FEATURE_COLUMNS = ['age', 'job', 'marital', 'education', 'default', 'balance',
                   'housing', 'loan', 'contact', 'day', 'month', 'duration',
                   'campaign', 'pdays', 'previous', 'poutcome']

def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    df_encoded = pd.get_dummies(df)
    for col in scaler.feature_names_in_:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[scaler.feature_names_in_]
    return scaler.transform(df_encoded)

def predict(data_dict):
    processed_data = preprocess_input(data_dict)
    prediction = model.predict(processed_data)[0]
    probability_np = model.predict_proba(processed_data)[0][1]
    probability = float(probability_np)
    return ("yes" if prediction == 1 else "no", probability)

@app.post("/predict/")
async def make_prediction(
    age: Annotated[int, Form(...)],
    job: Annotated[str, Form(...)],
    marital: Annotated[str, Form(...)],
    education: Annotated[str, Form(...)],
    default: Annotated[str, Form(...)],
    balance: Annotated[int, Form(...)],
    housing: Annotated[str, Form(...)],
    loan: Annotated[str, Form(...)],
    contact: Annotated[str, Form(...)],
    day: Annotated[int, Form(...)],
    month: Annotated[str, Form(...)],
    duration: Annotated[int, Form(...)],
    campaign: Annotated[int, Form(...)],
    pdays: Annotated[int, Form(...)],
    previous: Annotated[int, Form(...)],
    poutcome: Annotated[str, Form(...)],
):
    form_data = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
    }
    prediction_label, probability_value = predict(form_data)
    return {"prediction": prediction_label, "probability": f"{probability_value:.2f}"}

# To run this: uvicorn your_script_name:app --reload