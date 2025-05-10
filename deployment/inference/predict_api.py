"""
FastAPI scoring API for credit decisioning model.
"""

import os
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
from deployment.preprocessing.apply_pipeline import apply_feature_pipeline

# === Paths ===
MODEL_PATH = "deployment/model/credit_champion_model"
FEATURES_PATH = "deployment/model/selected_features.json"

# === Load model and feature list ===
model = load_model(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    selected_features = json.load(f)

# === FastAPI app ===
app = FastAPI(title="Credit Decisioning API")

# === Input schema ===
class ApplicationData(BaseModel):
    data: list  # list of dicts representing rows


@app.post("/predict")
def predict(app_data: ApplicationData):
    try:
        # Convert list of dicts to DataFrame
        df_raw = pd.DataFrame(app_data.data)

        # Apply preprocessing pipeline
        df_processed = apply_feature_pipeline(df_raw)

        # Select only trained features
        df_processed = df_processed[selected_features]

        # Predict
        preds = predict_model(model, data=df_processed)

        return {
            "predictions": preds["prediction_label"].tolist()
        }

    except Exception as e:
        return {"error": str(e)}
