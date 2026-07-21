from pathlib import Path

# ==========================
# Project Root
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================
# Data Directories
# ==========================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ==========================
# Models
# ==========================

MODEL_DIR = BASE_DIR / "models"

# ==========================
# Assets
# ==========================

ASSET_DIR = BASE_DIR / "assets"

# ==========================
# Main Dataset
# ==========================

FEATURED_DATASET = PROCESSED_DATA_DIR / "banking_fraud_featured.csv"

# ==========================
# Model Files
# ==========================

BEST_MODEL = MODEL_DIR / "best_model.pkl"

FEATURE_NAMES = MODEL_DIR / "feature_names.pkl"

LABEL_ENCODERS = MODEL_DIR / "label_encoders.pkl"

SHAP_IMPORTANCE = MODEL_DIR / "shap_feature_importance.csv"

MODEL_COMPARISON = MODEL_DIR / "model_comparison.csv"

XTRAIN = MODEL_DIR / "X_train.pkl"

XTEST = MODEL_DIR / "X_test.pkl"

YTRAIN = MODEL_DIR / "y_train.pkl"

YTEST = MODEL_DIR / "y_test.pkl"