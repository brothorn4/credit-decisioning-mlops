import pandas as pd

def encode_target(df, target_col='Default_Flag'):
    """
    Encode target column from 'Yes'/'No' to 1/0.

    INPUT:
    - df: DataFrame containing target column
    - target_col: Name of target column

    OUTPUT:
    - DataFrame with encoded target column (1 for 'Yes', 0 for 'No')
    """
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
    return df
