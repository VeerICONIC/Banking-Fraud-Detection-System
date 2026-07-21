from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from utils.config import FEATURED_DATASET

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(FEATURED_DATASET)

try:

    df = load_dataset()

    total_transactions = len(df)

    total_features = df.shape[1] - 1

    fraud_cases = int(df["isFraud"].sum())

    genuine_cases = total_transactions - fraud_cases

    fraud_rate = round(
        fraud_cases / total_transactions * 100,
        2
    )

except Exception:

    total_transactions = "N/A"

    total_features = "N/A"

    fraud_cases = "N/A"

    genuine_cases = "N/A"

    fraud_rate = "N/A"

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🛡️ Banking Fraud Detection System")

st.markdown("""
## Intelligent Fraud Detection using Machine Learning

This dashboard allows you to:

- 🔵 Detect fraudulent transactions

- 📊 Analyze banking transaction data

- 📈 Evaluate ML model performance

- 🧠 Understand predictions using SHAP

- ⚠️ Generate business risk scores
""")

st.divider()

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------

st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Transactions",
    total_transactions
)

c2.metric(
    "Features",
    total_features
)

c3.metric(
    "Fraud Cases",
    fraud_cases
)

c4.metric(
    "Fraud Rate",
    f"{fraud_rate}%"
)

st.divider()

# --------------------------------------------------
# Project Information
# --------------------------------------------------

st.subheader("📌 Project Information")

left, right = st.columns([2, 1])

with left:

    st.info(
"""
This project demonstrates an end-to-end Machine Learning pipeline for
detecting fraudulent banking transactions.

### Workflow

- Data Cleaning

- Feature Engineering

- Model Training

- Model Evaluation

- SHAP Explainability

- Risk Scoring

- FastAPI Deployment

- Interactive Streamlit Dashboard
"""
    )

with right:

    st.success(
"""
## Model Status

✅ Model Loaded

✅ Ready for Prediction

✅ SHAP Enabled

✅ Risk Scoring Enabled
"""
    )

st.divider()

# --------------------------------------------------
# Workflow
# --------------------------------------------------

st.subheader("⚙️ Workflow")

st.code(
"""
Raw Dataset

      ↓

Data Cleaning

      ↓

Feature Engineering

      ↓

Model Training

      ↓

Evaluation

      ↓

SHAP Explainability

      ↓

Risk Scoring

      ↓

FastAPI + Streamlit Dashboard
"""
)

st.divider()

# --------------------------------------------------
# Navigation
# --------------------------------------------------

st.subheader("🚀 Dashboard Modules")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
### 📊 Dataset Analysis

- Dataset Statistics

- Fraud Distribution

- Transaction Analysis
""")

    st.markdown("""
### 🔍 Single Prediction

- Predict one transaction

- Fraud Probability

- Risk Level
""")

    st.markdown("""
### 📈 Model Insights

- ROC Curve

- Confusion Matrix

- Feature Importance
""")

with col2:

    st.markdown("""
### 📂 Batch Prediction

- Upload CSV

- Predict Fraud

- Download Results
""")

    st.markdown("""
### 🧠 SHAP Explainability

- Summary Plot

- Waterfall Plot

- Feature Importance
""")

    st.markdown("""
### ⚠️ Risk Analysis

- Confidence

- Business Action

- Risk Category
""")

st.divider()

st.caption(
    "Developed using Scikit-Learn • XGBoost • SHAP • FastAPI • Streamlit"
)