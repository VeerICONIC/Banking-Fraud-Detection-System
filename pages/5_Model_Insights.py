from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.config import (
    MODEL_COMPARISON,
    SHAP_IMPORTANCE
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Model Insights",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Performance Insights")

st.markdown("""
Compare all Machine Learning models trained for fraud detection
and understand why the final model was selected.
""")

st.divider()

# --------------------------------------------------
# Load Data
# --------------------------------------------------

@st.cache_data
def load_model_results():
    return pd.read_csv(MODEL_COMPARISON)

@st.cache_data
def load_shap():
    return pd.read_csv(SHAP_IMPORTANCE)

comparison = load_model_results()
shap_df = load_shap()

# --------------------------------------------------
# Best Model
# --------------------------------------------------

best = comparison.sort_values(
    "ROC_AUC",
    ascending=False
).iloc[0]

st.success(
    f"""
🏆 Best Model Selected: **{best['Model']}**

ROC-AUC = **{best['ROC_AUC']:.4f}**

PR-AUC = **{best['PR_AUC']:.4f}**
"""
)

st.divider()

# --------------------------------------------------
# Metrics Table
# --------------------------------------------------

st.subheader("📊 Model Comparison")

display = comparison.copy()

for col in display.columns[1:]:

    display[col] = display[col].round(4)

st.dataframe(
    display,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# KPI Cards
# --------------------------------------------------

st.subheader("🏅 Best Model Metrics")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Accuracy",
    f"{best['Accuracy']:.4f}"
)

c2.metric(
    "Precision",
    f"{best['Precision']:.4f}"
)

c3.metric(
    "Recall",
    f"{best['Recall']:.4f}"
)

c4, c5, c6 = st.columns(3)

c4.metric(
    "F1 Score",
    f"{best['F1']:.4f}"
)

c5.metric(
    "ROC-AUC",
    f"{best['ROC_AUC']:.4f}"
)

c6.metric(
    "PR-AUC",
    f"{best['PR_AUC']:.4f}"
)

st.divider()

# --------------------------------------------------
# Accuracy Comparison
# --------------------------------------------------

st.subheader("📊 Accuracy Comparison")

fig = px.bar(
    comparison,
    x="Model",
    y="Accuracy",
    color="Model",
    text="Accuracy"
)

fig.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# Precision vs Recall
# --------------------------------------------------

st.subheader("🎯 Precision vs Recall")

metric_df = comparison.melt(
    id_vars="Model",
    value_vars=["Precision", "Recall"],
    var_name="Metric",
    value_name="Score"
)

fig = px.bar(
    metric_df,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text="Score"
)

fig.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# ROC-AUC Comparison
# --------------------------------------------------

st.subheader("📈 ROC-AUC Comparison")

fig = px.bar(
    comparison.sort_values(
        "ROC_AUC",
        ascending=False
    ),
    x="Model",
    y="ROC_AUC",
    color="ROC_AUC",
    text="ROC_AUC",
    color_continuous_scale="Blues"
)

fig.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig.update_layout(
    coloraxis_showscale=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# PR-AUC Comparison
# --------------------------------------------------

st.subheader("📊 Precision-Recall AUC")

fig = px.bar(
    comparison.sort_values(
        "PR_AUC",
        ascending=False
    ),
    x="Model",
    y="PR_AUC",
    color="PR_AUC",
    text="PR_AUC",
    color_continuous_scale="Greens"
)

fig.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig.update_layout(
    coloraxis_showscale=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# SHAP Feature Importance
# --------------------------------------------------

st.subheader("⭐ Top 20 Important Features")

top20 = (
    shap_df
    .sort_values(
        by=shap_df.columns[1],
        ascending=False
    )
    .head(20)
)

fig = px.bar(
    top20,
    x=top20.columns[1],
    y=top20.columns[0],
    orientation="h",
    text=round(top20.iloc[:, 1], 4),
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
# Model Ranking
# --------------------------------------------------

st.subheader("🏆 Overall Model Ranking")

ranking = comparison.sort_values(
    "ROC_AUC",
    ascending=False
).reset_index(drop=True)

ranking.index += 1

st.dataframe(
    ranking,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# Key Insights
# --------------------------------------------------

st.subheader("💡 Key Insights")

st.success(
"""
### Model Evaluation Summary

✅ **XGBoost** achieved the highest ROC-AUC (0.9554), indicating the best ability to distinguish fraudulent from legitimate transactions.

✅ **Random Forest** produced the highest PR-AUC, making it highly competitive on this imbalanced dataset.

✅ **LightGBM** delivered strong performance with fast training and competitive metrics.

✅ **Decision Tree** showed signs of overfitting and performed significantly worse.

✅ **Logistic Regression** served as a baseline model but underperformed compared to ensemble methods.

### Final Model Selection

The **XGBoost** classifier was selected as the production model due to its superior balance between Accuracy, Precision, Recall, F1 Score, ROC-AUC, and PR-AUC.
"""
)

st.sidebar.title("📈 Model Insights")

st.sidebar.success("Best Model: XGBoost")

st.sidebar.info(
"""
This page compares all trained models using:

• Accuracy

• Precision

• Recall

• F1 Score

• ROC-AUC

• PR-AUC

It also displays SHAP feature importance for model interpretability.
"""
)