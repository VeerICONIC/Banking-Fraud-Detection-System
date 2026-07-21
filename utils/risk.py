"""
Risk Scoring Utilities
----------------------
Contains helper functions for:
1. Risk Level Assignment
2. Recommended Action
3. Confidence Score
4. Confidence Level
"""

import numpy as np


def assign_risk(probability: float) -> str:
    """
    Assign risk category based on fraud probability.
    """

    if probability < 0.20:
        return "Low"

    elif probability < 0.50:
        return "Medium"

    elif probability < 0.80:
        return "High"

    return "Critical"


def recommend_action(risk_level: str) -> str:
    """
    Return recommended banking action.
    """

    actions = {
        "Low": "Approve Transaction",
        "Medium": "Manual Review",
        "High": "Hold Transaction",
        "Critical": "Block Transaction"
    }

    return actions.get(risk_level, "Unknown")


def calculate_confidence(probability: float, prediction: int) -> float:
    """
    Calculate confidence score of prediction.
    """

    if prediction == 1:
        return probability

    return 1 - probability


def confidence_level(confidence: float) -> str:
    """
    Convert confidence score into readable category.
    """

    if confidence >= 0.95:
        return "Very High"

    elif confidence >= 0.80:
        return "High"

    elif confidence >= 0.60:
        return "Medium"

    return "Low"


def generate_risk_report(probability: float, prediction: int) -> dict:
    """
    Generate complete fraud risk report.
    """

    risk = assign_risk(probability)

    confidence = calculate_confidence(
        probability,
        prediction
    )

    return {
        "Fraud Probability": round(probability * 100, 2),
        "Prediction": "Fraud" if prediction == 1 else "Legitimate",
        "Risk Level": risk,
        "Confidence": round(confidence * 100, 2),
        "Confidence Level": confidence_level(confidence),
        "Recommended Action": recommend_action(risk)
    }