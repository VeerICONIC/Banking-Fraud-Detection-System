from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from utils.preprocessing import prepare_features

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.pkl"

FEATURE_PATH = MODEL_DIR / "feature_names.pkl"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    feature_names = joblib.load(FEATURE_PATH)

    return model, feature_names


model, feature_names = load_model()


# --------------------------------------------------
# Align Features
# --------------------------------------------------

def align_features(df: pd.DataFrame):

    df = df.copy()

    for col in feature_names:

        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    return df


# --------------------------------------------------
# Predict Class
# --------------------------------------------------

def predict(df: pd.DataFrame):

    df = prepare_features(df)

    df = align_features(df)

    return model.predict(df)


# --------------------------------------------------
# Predict Probability
# --------------------------------------------------

def predict_probability(df: pd.DataFrame):

    df = prepare_features(df)

    df = align_features(df)

    return model.predict_proba(df)[:, 1]


# --------------------------------------------------
# Predict Transaction
# --------------------------------------------------

def predict_transaction(df: pd.DataFrame):

    df = prepare_features(df)

    df = align_features(df)

    prediction = int(model.predict(df)[0])

    probability = float(model.predict_proba(df)[0][1])

    return prediction, probability