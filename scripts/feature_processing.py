import pandas as pd
import numpy as np
import plotly.express as px

def analyze_features_iv(df, features, target='Default_Flag', bins=10, bin_method='quantile', show_plots=True):
    """
    Analyze a list of features and calculate their Information Values (IV).

    INPUT:
    - df: DataFrame with features and target variable
    - features: List of feature column names
    - target: Binary target column name
    - bins: Number of bins for binning
    - bin_method: Method for binning ('quantile' or 'uniform')
    - show_plots: Whether to display IV bar chart using Plotly

    OUTPUT:
    - iv_df: DataFrame with IV values per feature
    """
    iv_list = []

    for feature in features:
        try:
            if pd.api.types.is_numeric_dtype(df[feature]):
                if bin_method == 'quantile':
                    df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
                else:
                    df['bin'] = pd.cut(df[feature], bins=bins)
            else:
                df['bin'] = df[feature]
            
            grouped = df.groupby('bin', observed=False)[target].agg(['count', 'sum'])
            grouped['non_event'] = grouped['count'] - grouped['sum']
            grouped['event_rate'] = grouped['sum'] / grouped['sum'].sum()
            grouped['non_event_rate'] = grouped['non_event'] / grouped['non_event'].sum()
            
            grouped['woe'] = np.log((grouped['event_rate'] + 1e-10) / (grouped['non_event_rate'] + 1e-10))
            
            # Handle infinite or NaN WoE values
            if not np.isfinite(grouped['woe']).all():
                print(f"⚠️ WoE invalid values encountered for feature '{feature}' → replacing inf/NaN with 0.")
            grouped['woe'] = grouped['woe'].replace([np.inf, -np.inf], 0).fillna(0)
            
            grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']
            iv = grouped['iv'].sum()

            iv_list.append({'Feature': feature, 'IV': iv})
        except Exception as e:
            iv_list.append({'Feature': feature, 'IV': np.nan})
            print(f"Warning: Could not calculate IV for feature {feature}. Error: {e}")

    iv_df = pd.DataFrame(iv_list).sort_values(by='IV', ascending=False)

    if show_plots:
        fig = px.bar(iv_df, x='Feature', y='IV', title='Information Value by Feature')
        fig.show()

    return iv_df

def select_features_by_iv(iv_df, threshold=0.02):
    """
    Select features with IV >= threshold.

    INPUT:
    - iv_df: DataFrame with 'Feature' and 'IV' columns
    - threshold: Minimum IV threshold

    OUTPUT:
    - selected_features: List of features meeting threshold
    """
    selected_features = iv_df[iv_df['IV'] >= threshold]['Feature'].tolist()
    return selected_features
