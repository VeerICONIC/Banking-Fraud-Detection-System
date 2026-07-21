"""
Preprocessing Utilities
-----------------------
Handles preprocessing for inference.

Responsibilities:
1. Load label encoders
2. Encode categorical columns
3. Validate input
4. Prepare data for model prediction
"""

import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

ENCODER_PATH = MODEL_DIR / "label_encoders.pkl"

@st.cache_resource
def load_label_encoders():
    try:
        return joblib.load(ENCODER_PATH)
    except FileNotFoundError:
        return {}

label_encoders = load_label_encoders()


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic validation.
    """

    if df.empty:
        raise ValueError("Input data is empty.")

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values.

    Numeric -> Median
    Categorical -> Unknown
    """

    df = df.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(exclude=["number"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using saved label encoders.
    """

    df = df.copy()

    for col, encoder in label_encoders.items():

        if col not in df.columns:
            continue

        mapping = dict(zip(encoder.classes_,
                           encoder.transform(encoder.classes_)))

        df[col] = (
            df[col]
            .astype(str)
            .map(lambda x: mapping.get(x, -1))
        )

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    """

    df = validate_input(df)

    df = fill_missing_values(df)

    df = encode_features(df)

    return df