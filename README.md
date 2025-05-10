# 🧠 Credit Decisioning MLOps Project

This repository contains a modular, production-ready pipeline for building a credit decisioning machine learning model.

It follows best practices in data engineering, feature engineering, modeling, and evaluation within an MLOps-oriented framework.

---

## ✅ Current Pipeline Stages

This repo implements the following **pre-modeling, modeling, and batch deployment steps**:

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

### 7. Deployment Preparation
- Saved trained model to `deployment/model/credit_champion_model.pkl`
- Saved selected training features to `selected_features.json`
- Built preprocessing script `apply_pipeline.py` for raw input transformation
- Developed `predict_batch.py` to apply model and pipeline to new data
- Confirmed working prediction flow with real output to `outputs/predictions.csv`

---

## 📌 Next Phase (Planned)

The following modules are scoped for real-time deployment:
- FastAPI scoring service (`predict_api.py`)
- Input validation schema and Swagger test UI
- Enhanced error handling and logging
- Cloud-based endpoint hosting (e.g., AWS Lambda, API Gateway)
- Integration with monitoring and feedback loop for retraining

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
├── deployment/
│   ├── model/
│   │   ├── credit_champion_model.pkl
│   │   └── selected_features.json
│   ├── preprocessing/
│   │   └── apply_pipeline.py
│   ├── inference/
│   │   └── predict_batch.py
├── outputs/
│   └── predictions.csv
├── environment.yml
