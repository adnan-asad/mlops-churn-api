"""
preprocessing.py — Transform CustomerFeatures into model-ready feature vector.

Must match exactly the preprocessing used during training.
"""

import numpy as np
import pandas as pd
from app.schemas import CustomerFeatures


CONTRACT_MAP = {
    "Month-to-month": 0,
    "One year":        1,
    "Two year":        2,
}

INTERNET_MAP = {
    "No":          0,
    "DSL":         1,
    "Fiber optic": 2,
}

PAYMENT_MAP = {
    "Mailed check":             0,
    "Bank transfer (automatic)": 1,
    "Credit card (automatic)":  2,
    "Electronic check":         3,
}


def customer_to_features(customer: CustomerFeatures) -> np.ndarray:
    """
    Convert a CustomerFeatures Pydantic model into a numpy feature vector.

    Feature order must match the training pipeline exactly.

    Args:
        customer: Validated CustomerFeatures instance.

    Returns:
        1D numpy array of shape (13,) ready for model inference.
    """
    features = np.array([
        customer.tenure,
        customer.monthly_charges,
        customer.total_charges,
        CONTRACT_MAP.get(customer.contract, 0),
        INTERNET_MAP.get(customer.internet_service, 0),
        PAYMENT_MAP.get(customer.payment_method, 0),
        int(customer.paperless_billing),
        int(customer.senior_citizen),
        int(customer.partner),
        int(customer.dependents),
        int(customer.phone_service),
        int(customer.online_security),
        int(customer.tech_support),
    ], dtype=np.float32)

    return features.reshape(1, -1)


def get_risk_tier(probability: float) -> str:
    """
    Map a churn probability to a human-readable risk tier.

    Args:
        probability: Churn probability in [0, 1].

    Returns:
        "High", "Medium", or "Low".
    """
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


def get_confidence(probability: float) -> str:
    """
    Map a churn probability to a confidence label.

    Confidence is high when probability is far from 0.5 (decisive).

    Args:
        probability: Churn probability in [0, 1].

    Returns:
        "High", "Medium", or "Low".
    """
    distance = abs(probability - 0.5)
    if distance >= 0.3:
        return "High"
    elif distance >= 0.15:
        return "Medium"
    else:
        return "Low"
