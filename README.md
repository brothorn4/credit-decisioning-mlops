# Credit Decisioning MLOps Project

This repository contains a modular, production-ready pipeline for building a credit decisioning machine learning model.  
It follows best practices in data engineering, feature engineering, and model preparation within an MLOps framework.

## ✅ Current Pipeline Stages

This repo implements the following pre-modeling steps:

1. **EDA (Exploratory Data Analysis)**
   - Missing value analysis
   - Unique counts
   - Skewness and kurtosis checks
   - Correlation matrix and heatmap
   - Numeric feature distributions

2. **Preprocessing**
   - Encoding binary target variable (`Default_Flag`)

3. **Feature Engineering**
   - Creating binned features for `Utilization_Ratio`, `DTI_Ratio`, `Income`
   - Dropping raw numeric columns after binning
   - Automatically handling skewness for numeric features using Yeo-Johnson transform

4. **Feature Processing**
   - Calculating Information Value (IV) for all features
   - Handling invalid WoE values (e.g., infinite, NaN) inside the IV calculation
   - Selecting features based on IV threshold

5. **Train-Test Split**
   - Stratified train-test split with reproducible random state
   - Outputs ready-to-model `X_train`, `X_test`, `y_train`, `y_test`

## 📂 Project Structure

```plaintext
credit-decisioning-mlops/
├── scripts/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_processing.py
│   ├── data_split.py
│   ├── eda.py
├── notebooks/
│   ├── build_champion_model.ipynb
├── environment.yml
├── README.md
