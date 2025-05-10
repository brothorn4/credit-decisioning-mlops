"""
Apply preprocessing pipeline to new incoming data for scoring.
Includes: binning, skew handling, and feature selection ordering.
"""

import pandas as pd
import json
import os
from sklearn.preprocessing import PowerTransformer

# --- Configuration ---
BIN_METHOD = 'quantile'
BINS = 5
BIN_LABELS = list(range(BINS))

# Define columns involved in transformation
BIN_COLUMNS = ['Utilization_Ratio', 'DTI_Ratio', 'Income']
SKEW_COLUMNS = ['Months_Oldest_Trade', 'Avg_Deposit_Balance']
DROP_COLUMNS = BIN_COLUMNS

# Dynamically load selected features
def load_selected_features(path='deployment/model/selected_features.json'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"❌ selected_features.json not found at: {path}")

# --- Core transformations ---
def create_bins(df, column, bins=5, method='quantile', labels=None):
    if method == 'quantile':
        df[column + '_Bin'] = pd.qcut(df[column], q=bins, labels=labels, duplicates='drop')
    elif method == 'uniform':
        df[column + '_Bin'] = pd.cut(df[column], bins=bins, labels=labels)
    return df

def handle_skewness(df, columns):
    pt = PowerTransformer(method='yeo-johnson')
    for col in columns:
        try:
            reshaped = df[col].values.reshape(-1, 1)
            df[col] = pt.fit_transform(reshaped)
        except Exception:
            continue
    return df

# --- Main entry point ---
def apply_feature_pipeline(df_raw):
    df = df_raw.copy()

    # 1. Binning
    for col in BIN_COLUMNS:
        df = create_bins(df, col, bins=BINS, method=BIN_METHOD, labels=BIN_LABELS)

    # 2. Drop raw columns
    df.drop(columns=DROP_COLUMNS, errors='ignore', inplace=True)

    # 3. Apply skewness correction
    df = handle_skewness(df, SKEW_COLUMNS)

    # 4. Reorder to match trained model input
    selected_features = load_selected_features()
    df = df[[col for col in selected_features if col in df.columns]]

    return df
