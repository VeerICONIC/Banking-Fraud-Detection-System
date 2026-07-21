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

This application uses Machine Learning to detect fraudulent banking transactions,
assign risk levels, and explain model predictions using SHAP.

### Features
- 📊 Dataset Analysis
- 🔍 Single Transaction Prediction
- 📂 Batch CSV Prediction
- 📈 Model Performance
- 🧠 SHAP Explainability

Use the **sidebar** to navigate through the dashboard.
""")

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model", "Random Forest")
col2.metric("Task", "Fraud Detection")
col3.metric("Classes", "2")
col4.metric("Status", "🟢 Ready")

st.divider()

st.subheader("Project Workflow")

st.markdown(""")
""")

st.success("Navigate using the sidebar to explore different modules.")