import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

def create_binned_features(df, utilization_col='Utilization', dti_col='DTI', income_col='Income',
                           bins=5, bin_labels=None, bin_method='quantile', drop_original=True):
    """
    Create binned versions of Utilization, DTI, and Income columns, and optionally drop original columns.

    INPUT:
    - df: DataFrame with numeric columns
    - utilization_col: Name of Utilization column
    - dti_col: Name of DTI column
    - income_col: Name of Income column
    - bins: Number of bins to create
    - bin_labels: Optional list of labels for bins
    - bin_method: 'quantile' (default) or 'uniform'
    - drop_original: Whether to drop original columns after binning

    OUTPUT:
    - df: DataFrame with added binned columns (and optionally dropped raw columns)
    """
    if bin_labels is None:
        bin_labels = [f'Bin_{i+1}' for i in range(bins)]

    if bin_method == 'quantile':
        df[utilization_col + '_Bin'] = pd.qcut(df[utilization_col], q=bins, labels=bin_labels, duplicates='drop')
        df[dti_col + '_Bin'] = pd.qcut(df[dti_col], q=bins, labels=bin_labels, duplicates='drop')
        df[income_col + '_Bin'] = pd.qcut(df[income_col], q=bins, labels=bin_labels, duplicates='drop')
    else:
        df[utilization_col + '_Bin'] = pd.cut(df[utilization_col], bins=bins, labels=bin_labels)
        df[dti_col + '_Bin'] = pd.cut(df[dti_col], bins=bins, labels=bin_labels)
        df[income_col + '_Bin'] = pd.cut(df[income_col], bins=bins, labels=bin_labels)

    if drop_original:
        df = df.drop(columns=[utilization_col, dti_col, income_col])

    return df

def handle_skewness(df, exclude_cols=None, skew_threshold=1.0, method='yeo-johnson'):
    """
    Detect and correct skewness in numeric columns.

    INPUT:
    - df: DataFrame
    - exclude_cols: List of columns to exclude from transformation
    - skew_threshold: Absolute skew value above which to apply transform
    - method: 'yeo-johnson' (default) or 'log'

    OUTPUT:
    - df: DataFrame with transformed columns
    """
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    skewed_cols = []
    for col in numeric_cols:
        skew_val = df[col].skew()
        if abs(skew_val) >= skew_threshold:
            skewed_cols.append(col)
            print(f"Transforming {col} (skew={skew_val:.2f})")
    
    if skewed_cols:
        if method == 'yeo-johnson':
            pt = PowerTransformer(method='yeo-johnson')
            df[skewed_cols] = pt.fit_transform(df[skewed_cols])
        elif method == 'log':
            for col in skewed_cols:
                df[col] = np.log1p(df[col])
        else:
            raise ValueError("Invalid method. Choose 'yeo-johnson' or 'log'.")
    else:
        print("✅ No features exceeded skew threshold. No transformation applied.")

    return df
