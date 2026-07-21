from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.model import predict
from utils.model import predict_probability

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Batch Fraud Prediction")

st.markdown("""
Upload a CSV containing multiple transactions.
The model will predict fraud for every transaction and generate
a downloadable report.
""")

st.divider()

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Processed Feature Dataset (.csv)",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success(f"Loaded {len(df):,} transactions successfully.")

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        predict_btn = st.button(
            "🚀 Run Batch Prediction",
            use_container_width=True,
            type="primary"
        )

        if predict_btn:

            with st.spinner("Predicting transactions..."):

                prediction = predict(df)

                probability = predict_probability(df)

                result = df.copy()

                result["Prediction"] = prediction

                result["Prediction"] = result["Prediction"].map({
                    0: "Legitimate",
                    1: "Fraud"
                })

                result["Fraud Probability"] = (
                    probability * 100
                ).round(2)

                total = len(result)

                fraud = (
                    result["Prediction"] == "Fraud"
                ).sum()

                legitimate = total - fraud

                fraud_rate = round(
                    fraud / total * 100,
                    2
                )
                st.divider()

                st.subheader("Prediction Summary")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Total Transactions",
                    f"{total:,}"
                )

                c2.metric(
                    "Fraudulent",
                    f"{fraud:,}"
                )

                c3.metric(
                    "Legitimate",
                    f"{legitimate:,}"
                )

                c4.metric(
                    "Fraud Rate",
                    f"{fraud_rate}%"
                )

                st.divider()

                left, right = st.columns(2)

                with left:

                    st.subheader("📊 Prediction Distribution")

                    chart_df = pd.DataFrame(
                        {
                            "Prediction": [
                                "Legitimate",
                                "Fraud"
                            ],
                            "Count": [
                                legitimate,
                                fraud
                            ]
                        }
                    )

                    fig = px.bar(
                        chart_df,
                        x="Prediction",
                        y="Count",
                        text="Count",
                        color="Prediction",
                        color_discrete_sequence=[
                            "#2E8B57",
                            "#DC143C"
                        ]
                    )

                    fig.update_traces(
                        textposition="outside"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                with right:

                    st.subheader("🥧 Fraud Percentage")

                    pie = px.pie(
                        chart_df,
                        names="Prediction",
                        values="Count",
                        hole=0.45,
                        color="Prediction",
                        color_discrete_sequence=[
                            "#2E8B57",
                            "#DC143C"
                        ]
                    )

                    st.plotly_chart(
                        pie,
                        use_container_width=True
                    )

                st.divider()

                st.subheader("Prediction Results")

                st.dataframe(
                    result.head(25),
                    use_container_width=True,
                    height=450
                )

                csv = result.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.success(
                    "Batch prediction completed successfully!"
                )

    except Exception as e:

        st.error(f"Error: {e}")

else:

    st.info(
        "Upload a processed feature-engineered CSV file to begin batch prediction."
    )

st.sidebar.title("Batch Prediction")

st.sidebar.info(
"""
### Instructions

• Upload the processed feature-engineered dataset.

• The uploaded CSV should contain the same columns used during model training.

• The model predicts every transaction individually.

• Download the resulting CSV after prediction.

### Output Columns

- Prediction

- Fraud Probability
"""
)

st.sidebar.success("Model Status: Ready ✅")