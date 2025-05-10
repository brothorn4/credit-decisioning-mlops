from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
from pycaret.classification import load_model, predict_model
import json
import os
import uvicorn
import sys
from io import StringIO
from typing import List, Optional

# ✅ Ensure access to project root
sys.path.append(r"C:\Users\brook\credit-decisioning-mlops")
from deployment.preprocessing.apply_pipeline import apply_feature_pipeline

# --- Load model and feature schema ---
MODEL_PATH = "deployment/model/credit_champion_model"
FEATURES_PATH = "deployment/model/selected_features.json"

model = load_model(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    selected_features = json.load(f)

# --- FastAPI setup ---
app = FastAPI(title="Credit Decisioning Model API")

class Applicant(BaseModel):
    FICO_Score: int
    Num_Inquiries: int
    Utilization_Ratio: float
    Num_Tradelines: int
    Months_Oldest_Trade: int
    Derogs_30DPD: int
    Derogs_60DPD: int
    Derogs_90DPD: int
    Income: float
    DTI_Ratio: float
    Age: int
    Residence_Type: str
    Residence_Tenure: int
    Loan_Purpose: str
    Loan_Amount: int
    Loan_Term: int
    Is_Existing_Customer: int
    Relationship_Tenure: int
    Avg_Deposit_Balance: float
    Prev_Bank_Delinquency: int
    Unemployment_Rate: float
    Interest_Rate: float
    Consumer_Confidence: int
    Derog_Any: int
    Bureau_Thickness: str

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    error_rows: Optional[List[int]] = None
    error_messages: Optional[List[str]] = None

@app.get("/")
def health_check():
    return {"status": "API is up and running."}

@app.post("/predict")
def predict(applicant: Applicant):
    try:
        # Convert to DataFrame
        df_raw = pd.DataFrame([applicant.dict()])

        # Apply preprocessing pipeline
        df_processed = apply_feature_pipeline(df_raw)

        # Select model input columns
        df_selected = df_processed[[col for col in selected_features if col in df_processed.columns]]

        # Predict
        prediction = predict_model(model, data=df_selected)

        return {
            "prediction_label": int(prediction['prediction_label'].iloc[0]),
            "prediction_score": float(prediction['prediction_score'].iloc[0]) if 'prediction_score' in prediction else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Process a batch of applicants from a CSV file and return predictions.
    
    The CSV should contain columns matching the Applicant model fields.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read CSV content
        contents = await file.read()
        csv_content = contents.decode('utf-8')
        df_raw = pd.read_csv(StringIO(csv_content))
        
        # Track errors
        error_rows = []
        error_messages = []
        
        # Apply preprocessing pipeline
        try:
            df_processed = apply_feature_pipeline(df_raw)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error in preprocessing pipeline: {str(e)}")
        
        # Select model input columns
        required_columns = set(selected_features)
        available_columns = set(df_processed.columns)
        missing_columns = required_columns - available_columns
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        df_selected = df_processed[[col for col in selected_features if col in df_processed.columns]]
        
        # Predict
        predictions = predict_model(model, data=df_selected)
        
        # Format results
        results = []
        for idx, row in predictions.iterrows():
            try:
                # Add unique identifier if available in original data
                result = {
                    "row_index": idx,
                    "prediction_label": int(row['prediction_label']),
                    "prediction_score": float(row['prediction_score']) if 'prediction_score' in predictions.columns else None
                }
                
                # Include original data if needed
                # result["original_data"] = df_raw.iloc[idx].to_dict()
                
                results.append(result)
            except Exception as e:
                error_rows.append(idx)
                error_messages.append(f"Error processing row {idx}: {str(e)}")
        
        response = {"predictions": results}
        
        if error_rows:
            response["error_rows"] = error_rows
            response["error_messages"] = error_messages
            
        return response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

# Optional: Add endpoint to get feature schema
@app.get("/features")
def get_features():
    return {"required_features": selected_features}

if __name__ == "__main__":
    uvicorn.run("deployment.inference.predict_api:app", host="127.0.0.1", port=8000, reload=True)