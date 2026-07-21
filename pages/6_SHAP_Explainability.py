from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.config import SHAP_IMPORTANCE

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Model Explainability using SHAP")

st.markdown("""
Understand **why** the XGBoost model predicts a transaction as fraudulent.

This page uses **SHAP (SHapley Additive exPlanations)** to identify the most influential
features affecting fraud detection.
""")

st.divider()

# --------------------------------------------------
# Load SHAP Data
# --------------------------------------------------

@st.cache_data
def load_shap():
    return pd.read_csv(SHAP_IMPORTANCE)

shap_df = load_shap()

shap_df = shap_df.sort_values(
    by=shap_df.columns[1],
    ascending=False
)

# --------------------------------------------------
# Top Metrics
# --------------------------------------------------

st.subheader("📊 Explainability Summary")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Model",
    "XGBoost"
)

c2.metric(
    "Features Analysed",
    len(shap_df)
)

c3.metric(
    "Top Feature",
    shap_df.iloc[0, 0]
)

st.divider()

# --------------------------------------------------
# SHAP Importance
# --------------------------------------------------

st.subheader("⭐ Top 20 Most Important Features")

top20 = shap_df.head(20)

fig = px.bar(
    top20,
    x=top20.columns[1],
    y=top20.columns[0],
    orientation="h",
    text=round(top20.iloc[:,1],4),
    color=top20.columns[1],
    color_continuous_scale="Oranges"
)

fig.update_layout(
    height=700,
    yaxis=dict(autorange="reversed"),
    coloraxis_showscale=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# SHAP Values Table
# --------------------------------------------------

st.subheader("📋 SHAP Feature Importance Table")

display = top20.copy()

display.columns = [
    "Feature",
    "Mean |SHAP Value|"
]

st.dataframe(
    display,
    use_container_width=True,
    height=500
)

st.divider()

# --------------------------------------------------
# Top 10 Pie Chart
# --------------------------------------------------

st.subheader("🥧 Contribution of Top 10 Features")

top10 = shap_df.head(10)

fig = px.pie(
    top10,
    names=top10.columns[0],
    values=top10.columns[1],
    hole=0.45
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# Business Interpretation
# --------------------------------------------------

st.subheader("💡 Business Interpretation")

st.info(
"""
### What do SHAP values tell us?

SHAP assigns an importance value to every feature used by the model.

Features with larger SHAP values contribute more towards the model's prediction.

Positive influence generally pushes the prediction towards **Fraud** while lower influence contributes towards **Legitimate** classification.

Unlike traditional feature importance, SHAP provides consistent and theoretically sound explanations for every prediction.
"""
)

st.divider()

# --------------------------------------------------
# Why SHAP?
# --------------------------------------------------

st.subheader("📖 Why SHAP?")

st.success(
"""
✔ Model Agnostic Explainability

✔ Local and Global Interpretability

✔ Consistent Feature Attribution

✔ Helps Analysts understand Fraud Drivers

✔ Improves Trust in Machine Learning Predictions

✔ Widely adopted in Financial Services and Banking
"""
)

st.divider()

# --------------------------------------------------
# Fraud Detection Insights
# --------------------------------------------------

st.subheader("🏦 Fraud Detection Insights")

st.markdown("""
Based on SHAP analysis, the model primarily relies on a small subset of highly informative
features to distinguish fraudulent transactions.

The explainability analysis indicates that:

- High-value transactions contribute significantly to fraud prediction.

- Device and identity-related features provide strong discriminatory power.

- Card usage patterns and historical transaction behaviour are important fraud indicators.

- Email domain consistency and customer behavioural features improve detection accuracy.

These insights can help fraud analysts understand why the model flags particular transactions and support more informed investigation workflows.
""")

st.divider()

# --------------------------------------------------
# Top Features
# --------------------------------------------------

st.subheader("🏆 Top 5 Fraud Indicators")

for i in range(5):

    st.write(
        f"**{i+1}. {shap_df.iloc[i,0]}**"
    )

st.sidebar.title("🧠 SHAP Explainability")

st.sidebar.success("Explainability Enabled")

st.sidebar.info(
"""
This page explains model predictions using SHAP feature importance.

SHAP improves transparency and helps understand which variables influence fraud detection the most.
"""
)