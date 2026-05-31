"""
train_and_save_model.py — Generate a demo XGBoost model and save artifacts.

This script creates a synthetic dataset and trains an XGBoost classifier
so the API has a model to serve. In production, replace with a model
trained on real data.

Usage:
    python scripts/train_and_save_model.py
"""

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics         import roc_auc_score, accuracy_score
from xgboost                 import XGBClassifier


MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)


def generate_synthetic_data(n: int = 5000, seed: int = 42) -> tuple:
    """
    Generate a synthetic Telco-like churn dataset.

    Features match the CustomerFeatures schema exactly.
    Churn probability is a deterministic function of the features
    so the model learns meaningful patterns.

    Args:
        n:    Number of samples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y) numpy arrays.
    """
    rng = np.random.default_rng(seed)

    tenure           = rng.integers(0, 73,  size=n).astype(float)
    monthly_charges  = rng.uniform(20, 120, size=n)
    total_charges    = tenure * monthly_charges + rng.uniform(0, 50, size=n)
    contract         = rng.integers(0, 3,   size=n).astype(float)  # 0=M2M,1=1yr,2=2yr
    internet_service = rng.integers(0, 3,   size=n).astype(float)
    payment_method   = rng.integers(0, 4,   size=n).astype(float)
    paperless        = rng.integers(0, 2,   size=n).astype(float)
    senior           = rng.choice([0, 1], p=[0.84, 0.16], size=n).astype(float)
    partner          = rng.integers(0, 2,   size=n).astype(float)
    dependents       = rng.integers(0, 2,   size=n).astype(float)
    phone_service    = rng.integers(0, 2,   size=n).astype(float)
    online_security  = rng.integers(0, 2,   size=n).astype(float)
    tech_support     = rng.integers(0, 2,   size=n).astype(float)

    X = np.column_stack([
        tenure, monthly_charges, total_charges, contract,
        internet_service, payment_method, paperless, senior,
        partner, dependents, phone_service, online_security, tech_support,
    ])

    # Deterministic churn signal (matches real-world patterns)
    churn_score = (
        - 0.05 * tenure
        + 0.01 * monthly_charges
        - 0.8  * contract
        + 0.4  * (internet_service == 2)   # Fiber optic churns more
        + 0.3  * paperless
        - 0.2  * online_security
        - 0.2  * tech_support
        + rng.normal(0, 0.5, size=n)
    )
    y = (churn_score > 0).astype(int)
    return X, y


FEATURE_NAMES = [
    "tenure", "monthly_charges", "total_charges", "contract",
    "internet_service", "payment_method", "paperless_billing",
    "senior_citizen", "partner", "dependents",
    "phone_service", "online_security", "tech_support",
]


def main():
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(n=5000)
    print(f"Dataset: {X.shape} — Churn rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    print(f"Test Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

    # Save artifacts
    model_path = MODEL_DIR / "xgboost_model.joblib"
    names_path = MODEL_DIR / "feature_names.json"

    joblib.dump(model, model_path)
    names_path.write_text(json.dumps(FEATURE_NAMES))

    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Feature names saved to {names_path}")
    print("\nYou can now start the API:")
    print("  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
