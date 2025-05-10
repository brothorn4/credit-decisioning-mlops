import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_data(df, target_col='Default_Flag', test_size=0.2, random_state=42):
    """
    Split the data into train and test sets with stratification on target.

    INPUT:
    - df: Input DataFrame
    - target_col: Name of target column
    - test_size: Proportion for test set
    - random_state: Seed for reproducibility

    OUTPUT:
    - train_df: Training DataFrame
    - test_df: Testing DataFrame
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )
    return train_df, test_df
