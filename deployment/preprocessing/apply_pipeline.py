"""
Apply preprocessing pipeline to new incoming data for scoring.
Includes: binning, skew handling, and feature ordering.
"""

import pandas as pd
from sklearn.preprocessing import PowerTransformer

# --- Load config (can be customized to load from JSON) ---
BIN_METHOD = 'quantile'
BINS = 5
BIN_LABELS = list(range(BINS))
SKEW_COLUMNS = ['Months_Oldest_Trade', 'Avg_Deposit_Balance']
BIN_COLUMNS = ['Utilization_Ratio', 'DTI_Ratio', 'Income']
DROP_COLUMNS = BIN_COLUMNS
FEATURE_ORDER = []  # Load from selected_features.json in practice


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


def apply_feature_pipeline(df_raw):
    """
    Main callable for scoring pipeline. Returns processed DataFrame.
    """
    df = df_raw.copy()

    # Create binned features
    for col in BIN_COLUMNS:
        df = create_bins(df, col, bins=BINS, method=BIN_METHOD, labels=BIN_LABELS)

    # Drop raw features after binning
    df.drop(columns=DROP_COLUMNS, errors='ignore', inplace=True)

    # Apply skewness correction
    df = handle_skewness(df, SKEW_COLUMNS)

    # Reorder features if FEATURE_ORDER is provided
    if FEATURE_ORDER:
        df = df[[col for col in FEATURE_ORDER if col in df.columns]]

    return df
