import streamlit as st

st.set_page_config(
    page_title="Banking Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Banking Fraud Detection System")

st.markdown("""
Welcome to the **Banking Fraud Detection Dashboard**.

This dashboard demonstrates an end-to-end Machine Learning pipeline for fraud detection.

### Available Modules
- 🏠 Home
- 📊 Dataset Analysis
- 🔍 Single Prediction
- 📂 Batch Prediction
- 📈 Model Insights
- 🧠 SHAP Explainability

👈 Select a page from the sidebar to begin.
""")

st.info("Use the navigation panel on the left to explore the dashboard.")