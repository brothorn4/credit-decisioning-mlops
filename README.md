# Credit Decisioning MLOps Project

This repository contains a modular, production-ready pipeline for building and deploying a credit decisioning machine learning model using PyCaret and FastAPI.

It follows best practices in data engineering, feature engineering, model preparation, and scoring within an MLOps framework.

---

## ✅ Current Pipeline Stages

### 1. EDA (Exploratory Data Analysis)
- Missing value analysis
- Skewness and kurtosis checks
- Correlation matrix and heatmap
- Numeric feature distributions

### 2. Preprocessing
- Encoding binary target variable (`Default_Flag`)
- Cleaning or flagging invalid entries

### 3. Feature Engineering
- Creating binned features (`Utilization_Ratio`, `DTI_Ratio`, `Income`)
- Dropping raw numeric columns after binning
- Handling skewness with Yeo-Johnson transform

### 4. Feature Processing
- Calculating Information Value (IV)
- Selecting features based on IV threshold
- Returning selected feature list as JSON

### 5. Modeling
- PyCaret model training and tuning
- Model export to `.pkl`
- Selected features exported to `.json`

### 6. Batch Scoring
- Run `predict_batch.py` on new files
- Preprocessing pipeline reused before scoring
- Conda Prompt Environment

### 7. FastAPI Inference
- Exposes `/predict` endpoint
- Accepts raw csv and returns prediction + score
- Uses same model and preprocessing logic as batch

---

## 🚀 How to Use

### 🔁 Batch Prediction via Python

```bash
conda activate pycaret_cls_env
python deployment/inference/predict_batch.py
