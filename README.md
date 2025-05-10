# 🧠 Credit Decisioning MLOps Project

This repository contains a modular, production-ready pipeline for building a credit decisioning machine learning model.

It follows best practices in data engineering, feature engineering, modeling, and evaluation within an MLOps-oriented framework.

---

## ✅ Current Pipeline Stages

This repo implements the following **pre-modeling and modeling steps**:

### 1. EDA (Exploratory Data Analysis)
- Missing value analysis
- Unique value distributions
- Skewness and kurtosis checks
- Correlation matrix and heatmap
- Numeric feature distributions

### 2. Preprocessing
- Encoding binary target variable (`Default_Flag`)

### 3. Feature Engineering
- Creating binned features for `Utilization_Ratio`, `DTI_Ratio`, `Income`
- Dropping raw numeric columns after binning
- Automatically handling skewness using Yeo-Johnson transform

### 4. Feature Processing
- Calculating Information Value (IV) for all features
- Handling invalid WoE values (e.g., infinite, NaN) inside IV calculation
- Selecting features based on IV thresholds

### 5. Train-Test Split
- Stratified split of data into training and test sets
- Optional seed control for reproducibility

### 6. Model Build (Champion Model)
- PyCaret-based setup, model comparison, and selection
- Hyperparameter tuning with `tune_model()`
- Model finalization using holdout test set
- Evaluation using:
  - Confusion matrix
  - Classification report (sklearn)
  - Sankey error flow diagram (custom Plotly)

---

## 📌 Next Phase (Planned)

The following modules are planned for the upcoming deployment milestone:
- Model artifact packaging
- Preprocessing pipeline wrapper for new data
- Batch and real-time inference scripts
- Deployment using AWS/FastAPI

---

## 📁 Directory Structure (So Far)

```bash
.
├── notebooks/
│   └── build_champion_model.ipynb
├── scripts/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_processing.py
│   └── data_split.py
├── environment.yml
