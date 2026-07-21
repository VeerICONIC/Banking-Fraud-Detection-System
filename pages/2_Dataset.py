from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.config import FEATURED_DATASET, SHAP_IMPORTANCE

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Dataset Analysis",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(FEATURED_DATASET)

@st.cache_data
def load_shap():
    return pd.read_csv(SHAP_IMPORTANCE)

df = load_dataset()

try:
    shap_df = load_shap()
except:
    shap_df = None

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📊 Dataset Analysis")

st.markdown(
"""
Explore the processed IEEE-CIS Fraud Detection dataset used for
training the Machine Learning model.
"""
)

st.divider()

# --------------------------------------------------
# Dataset Summary
# --------------------------------------------------

rows = len(df)
cols = df.shape[1]
fraud = int(df["isFraud"].sum())
legitimate = rows - fraud
fraud_rate = round((fraud / rows) * 100, 2)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Transactions", f"{rows:,}")
c2.metric("Features", cols)
c3.metric("Fraud Cases", f"{fraud:,}")
c4.metric("Fraud Rate", f"{fraud_rate}%")

st.divider()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True,
    height=350
)

st.divider()

# --------------------------------------------------
# Class Distribution
# --------------------------------------------------

left, right = st.columns(2)

with left:

    st.subheader("🥧 Fraud Distribution")

    fig = px.pie(
        names=["Legitimate", "Fraud"],
        values=[legitimate, fraud],
        hole=0.5,
        color_discrete_sequence=["#4CAF50", "#F44336"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    st.subheader("📈 Class Distribution")

    class_df = pd.DataFrame({
        "Class": ["Legitimate", "Fraud"],
        "Count": [legitimate, fraud]
    })

    fig = px.bar(
        class_df,
        x="Class",
        y="Count",
        text="Count",
        color="Class",
        color_discrete_sequence=["#4CAF50", "#F44336"]
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# --------------------------------------------------
# Transaction Amount Distribution
# --------------------------------------------------

if "TransactionAmt" in df.columns:

    st.subheader("💰 Transaction Amount Distribution")

    fig = px.histogram(
        df,
        x="TransactionAmt",
        nbins=60,
        color_discrete_sequence=["#1976D2"]
    )

    fig.update_layout(
        xaxis_title="Transaction Amount",
        yaxis_title="Frequency"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# --------------------------------------------------
# SHAP Feature Importance
# --------------------------------------------------

if shap_df is not None:

    st.subheader("🧠 Top 20 Important Features (SHAP)")

    shap_df = shap_df.sort_values(
        by=shap_df.columns[1],
        ascending=False
    ).head(20)

    fig = px.bar(
        shap_df,
        x=shap_df.columns[1],
        y=shap_df.columns[0],
        orientation="h",
        text=round(shap_df.iloc[:,1],4),
        color=shap_df.columns[1],
        color_continuous_scale="Blues"
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
# Dataset Information
# --------------------------------------------------

st.subheader("📌 Dataset Information")

info = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(
    info,
    use_container_width=True,
    height=450
)

st.divider()

st.success("Dataset loaded successfully and ready for prediction.")