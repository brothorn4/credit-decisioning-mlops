"""
Batch scoring script for the credit decisioning model.
"""

import os
import sys
import pandas as pd
from pycaret.classification import load_model, predict_model

# ✅ Add project root to Python path to enable cross-folder imports
sys.path.append(r"C:\Users\brook\credit-decisioning-mlops")

from deployment.preprocessing.apply_pipeline import apply_feature_pipeline

# 🔧 Define paths
MODEL_PATH = r"C:\Users\brook\credit-decisioning-mlops\deployment\model\credit_champion_model"
INPUT_PATH = r"C:\Users\brook\credit-decisioning-mlops\data\new_applications.csv"
OUTPUT_PATH = r"C:\Users\brook\credit-decisioning-mlops\outputs\predictions.csv"

def run_batch_prediction(input_path, output_path):
    print(f"\n📥 Loading input data from: {input_path}")
    df_raw = pd.read_csv(input_path)

    print(f"🧼 Running preprocessing pipeline...")
    df_processed = apply_feature_pipeline(df_raw)

    print(f"📦 Loading trained model...")
    model = load_model(MODEL_PATH)

    print(f"🔮 Predicting...")
    df_pred = predict_model(model, data=df_processed)

    print(f"💾 Saving results to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_pred.to_csv(output_path, index=False)

    print(f"\n✅ Done.")

# 🔁 Entry point
if __name__ == "__main__":
    run_batch_prediction(INPUT_PATH, OUTPUT_PATH)
