"""
model.py — Model loading and inference utilities.

Loads the trained XGBoost model from disk at startup
and exposes a predict function for the API routes.
"""

import os
import joblib
import numpy as np
from pathlib import Path


MODEL_PATH = os.getenv("MODEL_PATH", "model/xgboost_model.joblib")
_model     = None


def load_model():
    """
    Load the XGBoost model from disk into the global _model variable.

    Called once at application startup.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    global _model
    path = Path(MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run: python scripts/train_and_save_model.py"
        )
    _model = joblib.load(path)
    print(f"Model loaded from {path}")


def is_model_loaded() -> bool:
    """Return True if the model has been loaded successfully."""
    return _model is not None


def predict_proba(features: np.ndarray) -> float:
    """
    Run inference and return the churn probability.

    Args:
        features: numpy array of shape (1, n_features).

    Returns:
        Churn probability as a float in [0, 1].

    Raises:
        RuntimeError: If the model has not been loaded.
    """
    if _model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")
    proba = _model.predict_proba(features)[0][1]
    return float(proba)
