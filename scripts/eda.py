import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda_summary(df, target_col='Default_Flag', show_plots=True):
    """
    Generate an EDA summary for the input DataFrame.

    INPUT:
    - df: DataFrame
    - target_col: Name of target column
    - show_plots: Whether to display plots

    OUTPUT:
    - Prints key summaries
    - Returns correlation matrix
    """
    print(f"✅ Data shape: {df.shape}")
    
    print("\n🔍 Missing values per column:")
    print(df.isnull().sum())
    
    print("\n🔍 Unique counts per column:")
    print(df.nunique())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n🔍 Numeric feature summary:")
    print(df[numeric_cols].describe().T)
    
    print("\n🔍 Skewness:")
    print(df[numeric_cols].skew())
    
    print("\n🔍 Kurtosis:")
    print(df[numeric_cols].kurtosis())
    
    if target_col in df.columns:
        print("\n🔍 Target distribution:")
        print(df[target_col].value_counts(normalize=True))
    
    # Correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    if show_plots:
        plt.figure(figsize=(12,8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        
        for col in numeric_cols:
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
            fig.show()
    
    return corr_matrix
