from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import joblib

from utils.config import FEATURED_DATASET
from utils.model import predict_transaction
from utils.risk import generate_risk_report

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Single Prediction",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Single Transaction Fraud Prediction")

st.markdown(
"""
Predict whether a banking transaction is **Fraudulent** or **Legitimate**
using the trained Machine Learning model.
"""
)

st.divider()

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(FEATURED_DATASET)

try:
    df = load_dataset()

except Exception as e:

    st.error(f"Unable to load dataset.\n\n{e}")
    st.stop()

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "sample_df" not in st.session_state:
    st.session_state.sample_df = None

# --------------------------------------------------
# Select Input Method
# --------------------------------------------------

st.subheader("Choose Input Method")

left, right = st.columns(2)

with left:

    random_btn = st.button(
        "🎲 Load Random Transaction",
        use_container_width=True
    )

with right:

    uploaded_file = st.file_uploader(
        "Upload Single Row CSV",
        type=["csv"]
    )

# --------------------------------------------------
# Random Sample
# --------------------------------------------------

if random_btn:

    sample = df.sample(
        n=1,
        random_state=None
    )

    st.session_state.sample_df = sample.drop(
        columns=["isFraud"],
        errors="ignore"
    )

# --------------------------------------------------
# Uploaded CSV
# --------------------------------------------------

if uploaded_file is not None:

    try:

        upload_df = pd.read_csv(uploaded_file)

        if len(upload_df) != 1:

            st.error(
                "Please upload a CSV containing exactly ONE transaction."
            )

        else:

            st.session_state.sample_df = upload_df

    except Exception as e:

        st.error(e)

# --------------------------------------------------
# Show Selected Transaction
# --------------------------------------------------

if st.session_state.sample_df is not None:

    st.divider()

    st.subheader("Selected Transaction")

    st.dataframe(
        st.session_state.sample_df,
        use_container_width=True
    )

    predict = st.button(
        "🚀 Predict Fraud",
        use_container_width=True,
        type="primary"
    )

    if predict:

        with st.spinner("Predicting..."):

            try:

                prediction, probability = predict_transaction(
                    st.session_state.sample_df
                )

                report = generate_risk_report(
                        probability,
                        prediction
                        )

                fraud_probability = probability * 100

                confidence = report["Confidence"]

                confidence_level = report["Confidence Level"]

                risk_level = report["Risk Level"]

                action = report["Recommended Action"]

                prediction_label = report["Prediction"]
                
                st.divider()

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:

                    if prediction == 1:

                        st.error("## 🚨 Fraudulent Transaction")

                    else:

                        st.success("## ✅ Legitimate Transaction")

                    st.metric(
                        "Fraud Probability",
                        f"{fraud_probability:.2f}%"
                    )

                    st.metric(
                        "Risk Level",
                        risk_level
                    )

                with col2:

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.metric(
                        "Confidence Level",
                        confidence_level
                    )

                    st.info(
                        f"**Recommended Action:**\n\n{action}"
                    )

                st.divider()

                st.subheader("Prediction Summary")

                summary = pd.DataFrame({
                    "Prediction": [prediction_label],
                    "Fraud Probability (%)": [round(fraud_probability, 2)],
                    "Risk Level": [risk_level],
                    "Confidence (%)": [round(confidence, 2)],
                    "Confidence Level": [confidence_level],
                    "Recommended Action": [action]
                })

                st.dataframe(
                    summary,
                    use_container_width=True
                )

                csv = summary.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "📥 Download Prediction Report",
                    data=csv,
                    file_name="prediction_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:

                st.error(f"Prediction Failed\n\n{e}")

# --------------------------------------------------
# Sidebar Information
# --------------------------------------------------

st.sidebar.title("ℹ️ Instructions")

st.sidebar.info(
"""
### Random Transaction

Loads a random transaction from the processed dataset.

---

### Upload CSV

Upload a CSV containing exactly **one transaction**.

The uploaded CSV should contain the same feature columns that were used for model training.

---

### Prediction Output

- Fraud Probability
- Risk Level
- Confidence Score
- Recommended Business Action
"""
)

st.sidebar.success("Model Status: Ready ✅")