#!/usr/bin/env python3
"""
Run this script once from inside the mlops-churn-api folder.
It creates the entire project structure with all files.

Usage:
    python setup_project.py
"""

import os
from pathlib import Path

files = {}

# ─── .gitignore ────────────────────────────────────────────────────────────
files[".gitignore"] = """__pycache__/
*.py[cod]
*.pyo
venv/
env/
.env
*.egg-info/
dist/
build/
.DS_Store
Thumbs.db
.vscode/
.idea/
outputs/
model/*.joblib
model/*.pkl
"""

# ─── requirements.txt ──────────────────────────────────────────────────────
files["requirements.txt"] = """fastapi==0.110.0
uvicorn[standard]==0.29.0
pydantic==2.6.4
scikit-learn==1.4.1
xgboost==2.0.3
imbalanced-learn==0.12.0
pandas==2.2.1
numpy==1.26.4
joblib==1.3.2
pytest==8.1.0
httpx==0.27.0
"""

# ─── README.md ─────────────────────────────────────────────────────────────
files["README.md"] = """# ⚙️ MLOps Churn Prediction API

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)
[![CI](https://github.com/adnan-asad/mlops-churn-api/actions/workflows/ci.yml/badge.svg)](https://github.com/adnan-asad/mlops-churn-api/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready REST API that serves a customer churn prediction model.
Demonstrates end-to-end MLOps: model training, FastAPI serving, Docker containerisation,
and automated CI/CD with GitHub Actions.

Built as part of an M.Sc. AI & Data Science portfolio at Deggendorf Institute of Technology.

---

## Architecture

```
┌─────────────┐     POST /predict      ┌──────────────────────┐
│   Client    │ ──────────────────────► │   FastAPI App        │
│ (curl/app)  │                         │                      │
└─────────────┘ ◄────────────────────── │  ┌────────────────┐  │
                  {churn_probability,   │  │  XGBoost Model │  │
                   churn_label,         │  │  + Preprocessor│  │
                   risk_tier}           │  └────────────────┘  │
                                        └──────────────────────┘
                                                  │
                                        ┌─────────▼──────────┐
                                        │   Docker Container  │
                                        │   Port 8000         │
                                        └─────────────────────┘
                                                  │
                                        ┌─────────▼──────────┐
                                        │  GitHub Actions CI  │
                                        │  test → lint →      │
                                        │  build Docker       │
                                        └─────────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — returns model status |
| POST | `/predict` | Predict churn for one customer |
| POST | `/predict/batch` | Predict churn for multiple customers |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/redoc` | ReDoc API documentation |

---

## Quick Start

### Option A — Python (local)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/train_and_save_model.py   # Train and save the model
uvicorn app.main:app --reload --port 8000
```

### Option B — Docker
```bash
python scripts/train_and_save_model.py   # Train model first
docker-compose up
```

Then open: http://localhost:8000/docs

---

## API Usage Examples

### Health check
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model": "loaded", "version": "1.0.0"}
```

### Single prediction
```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "tenure": 2,
    "monthly_charges": 79.85,
    "total_charges": 159.70,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "paperless_billing": true,
    "senior_citizen": false,
    "partner": false,
    "dependents": false,
    "phone_service": true,
    "online_security": false,
    "tech_support": false
  }'
```
```json
{
  "churn_probability": 0.847,
  "churn_label": "Churn",
  "risk_tier": "High",
  "confidence": "High"
}
```

### Batch prediction
```bash
curl -X POST http://localhost:8000/predict/batch \\
  -H "Content-Type: application/json" \\
  -d '{"customers": [{ ...customer1... }, { ...customer2... }]}'
```

---

## Running Tests
```bash
pytest tests/ -v
```

---

## CI/CD Pipeline

Every push to `main` triggers GitHub Actions:
1. ✅ Install dependencies
2. ✅ Run pytest (all endpoints + preprocessing)
3. ✅ Flake8 lint check
4. ✅ Build Docker image

See `.github/workflows/ci.yml` for the full pipeline definition.

---

## Project Structure

```
mlops-churn-api/
├── app/
│   ├── main.py           # FastAPI app, routes
│   ├── model.py          # Model loading and inference
│   ├── schemas.py        # Pydantic input/output schemas
│   └── preprocessing.py  # Feature transformation pipeline
├── model/
│   └── README.md         # Model artifact instructions
├── scripts/
│   └── train_and_save_model.py  # Train XGBoost + save artifacts
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Future Work
- Add model versioning with MLflow
- Deploy to AWS Lambda or Google Cloud Run
- Add A/B testing between model versions
- Prometheus metrics endpoint for monitoring

---

## License
MIT License — see [LICENSE](LICENSE)

---
*Part of the GitHub portfolio of Asadullah Adnan — M.Sc. AI & Data Science, THD*
"""

# ─── LICENSE ───────────────────────────────────────────────────────────────
files["LICENSE"] = """MIT License

Copyright (c) 2025 Asadullah Adnan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""

# ─── Dockerfile ────────────────────────────────────────────────────────────
files["Dockerfile"] = """FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY model/ ./model/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# ─── docker-compose.yml ────────────────────────────────────────────────────
files["docker-compose.yml"] = """version: '3.8'

services:
  churn-api:
    build: .
    container_name: churn-prediction-api
    ports:
      - "8000:8000"
    volumes:
      - ./model:/app/model
    environment:
      - MODEL_PATH=/app/model/xgboost_model.joblib
      - PREPROCESSOR_PATH=/app/model/preprocessor.joblib
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""

# ─── .github/workflows/ci.yml ──────────────────────────────────────────────
files[".github/workflows/ci.yml"] = """name: CI Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-and-lint:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install flake8

      - name: Train model for tests
        run: python scripts/train_and_save_model.py

      - name: Run pytest
        run: pytest tests/ -v --tb=short

      - name: Lint with flake8
        run: |
          flake8 app/ scripts/ --max-line-length=100 --ignore=E501,W503

  build-docker:
    runs-on: ubuntu-latest
    needs: test-and-lint

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Train model for Docker build
        run: |
          pip install -r requirements.txt
          python scripts/train_and_save_model.py

      - name: Build Docker image
        run: docker build -t churn-api:latest .

      - name: Test Docker container starts
        run: |
          docker run -d -p 8000:8000 --name test-container churn-api:latest
          sleep 5
          curl -f http://localhost:8000/health
          docker stop test-container
"""

# ─── app/__init__.py ───────────────────────────────────────────────────────
files["app/__init__.py"] = '"""MLOps Churn API — app package."""\n'

# ─── app/schemas.py ────────────────────────────────────────────────────────
files["app/schemas.py"] = '''"""
schemas.py — Pydantic request and response models.

Defines strict input validation and typed API responses.
"""

from pydantic import BaseModel, Field, field_validator
from typing   import List, Optional


class CustomerFeatures(BaseModel):
    """Input schema for a single customer churn prediction request."""

    tenure:            int   = Field(..., ge=0, le=72,
                                     description="Months as customer (0-72)")
    monthly_charges:   float = Field(..., ge=0,
                                     description="Monthly bill in USD")
    total_charges:     float = Field(..., ge=0,
                                     description="Total amount billed in USD")
    contract:          str   = Field(...,
                                     description="Month-to-month | One year | Two year")
    internet_service:  str   = Field(...,
                                     description="DSL | Fiber optic | No")
    payment_method:    str   = Field(...,
                                     description="Electronic check | Mailed check | "
                                                 "Bank transfer (automatic) | "
                                                 "Credit card (automatic)")
    paperless_billing: bool  = Field(..., description="Whether on paperless billing")
    senior_citizen:    bool  = Field(False, description="Whether senior citizen")
    partner:           bool  = Field(False, description="Has a partner")
    dependents:        bool  = Field(False, description="Has dependents")
    phone_service:     bool  = Field(True,  description="Has phone service")
    online_security:   bool  = Field(False, description="Has online security add-on")
    tech_support:      bool  = Field(False, description="Has tech support add-on")

    @field_validator("contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        valid = {"Month-to-month", "One year", "Two year"}
        if v not in valid:
            raise ValueError(f"contract must be one of: {valid}")
        return v

    @field_validator("internet_service")
    @classmethod
    def validate_internet(cls, v: str) -> str:
        valid = {"DSL", "Fiber optic", "No"}
        if v not in valid:
            raise ValueError(f"internet_service must be one of: {valid}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 2,
                "monthly_charges": 79.85,
                "total_charges": 159.70,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
                "paperless_billing": True,
                "senior_citizen": False,
                "partner": False,
                "dependents": False,
                "phone_service": True,
                "online_security": False,
                "tech_support": False,
            }
        }
    }


class ChurnPrediction(BaseModel):
    """Response schema for a single churn prediction."""

    churn_probability: float  = Field(..., description="Probability of churn (0-1)")
    churn_label:       str    = Field(..., description="Churn | No Churn")
    risk_tier:         str    = Field(..., description="High | Medium | Low")
    confidence:        str    = Field(..., description="High | Medium | Low")


class BatchRequest(BaseModel):
    """Request schema for batch prediction."""
    customers: List[CustomerFeatures]


class BatchPrediction(BaseModel):
    """Response schema for batch prediction."""
    predictions:    List[ChurnPrediction]
    total:          int
    churn_count:    int
    churn_rate:     float


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status:  str
    model:   str
    version: str
'''

# ─── app/preprocessing.py ──────────────────────────────────────────────────
files["app/preprocessing.py"] = '''"""
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
'''

# ─── app/model.py ──────────────────────────────────────────────────────────
files["app/model.py"] = '''"""
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
'''

# ─── app/main.py ───────────────────────────────────────────────────────────
files["app/main.py"] = '''"""
main.py — FastAPI application with churn prediction endpoints.

Endpoints:
  GET  /health          — Health check
  POST /predict         — Single customer prediction
  POST /predict/batch   — Batch prediction
  GET  /docs            — Swagger UI (auto-generated)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas       import (CustomerFeatures, ChurnPrediction,
                                BatchRequest, BatchPrediction, HealthResponse)
from app.preprocessing import customer_to_features, get_risk_tier, get_confidence
from app import model as model_module

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"


# ── Lifespan: load model on startup ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model...")
    try:
        model_module.load_model()
        logger.info("Model loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Model load failed: {e}")
    yield
    logger.info("Shutting down.")


# ── App factory ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Production ML API for customer churn prediction.\\n\\n"
        "Built with FastAPI + XGBoost. Part of the MLOps portfolio of "
        "Asadullah Adnan — M.Sc. AI & Data Science, THD."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns model load status and API version.
    Always returns 200 so Docker HEALTHCHECK passes.
    """
    return HealthResponse(
        status  = "ok",
        model   = "loaded" if model_module.is_model_loaded() else "not loaded",
        version = APP_VERSION,
    )


@app.post("/predict", response_model=ChurnPrediction, tags=["Prediction"])
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Returns churn probability, binary label, risk tier, and confidence level.
    Input is validated automatically by Pydantic — invalid fields return 422.
    """
    if not model_module.is_model_loaded():
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Try again shortly.")

    features    = customer_to_features(customer)
    probability = model_module.predict_proba(features)
    churn_label = "Churn" if probability >= 0.5 else "No Churn"

    logger.info(
        f"Prediction: tenure={customer.tenure} "
        f"contract={customer.contract} "
        f"prob={probability:.3f} label={churn_label}"
    )

    return ChurnPrediction(
        churn_probability = round(probability, 4),
        churn_label       = churn_label,
        risk_tier         = get_risk_tier(probability),
        confidence        = get_confidence(probability),
    )


@app.post("/predict/batch", response_model=BatchPrediction, tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """
    Predict churn for a list of customers in a single request.

    Returns individual predictions plus aggregate statistics
    (total count, churn count, churn rate).
    """
    if not model_module.is_model_loaded():
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Try again shortly.")

    predictions = []
    for customer in request.customers:
        features    = customer_to_features(customer)
        probability = model_module.predict_proba(features)
        churn_label = "Churn" if probability >= 0.5 else "No Churn"
        predictions.append(ChurnPrediction(
            churn_probability = round(probability, 4),
            churn_label       = churn_label,
            risk_tier         = get_risk_tier(probability),
            confidence        = get_confidence(probability),
        ))

    churn_count = sum(1 for p in predictions if p.churn_label == "Churn")
    logger.info(f"Batch prediction: {len(predictions)} customers, "
                f"{churn_count} churns ({churn_count/len(predictions):.1%})")

    return BatchPrediction(
        predictions  = predictions,
        total        = len(predictions),
        churn_count  = churn_count,
        churn_rate   = round(churn_count / len(predictions), 4),
    )
'''

# ─── model/README.md ───────────────────────────────────────────────────────
files["model/README.md"] = """# Model Artifacts

This directory stores the trained model and preprocessor.
These files are **git-ignored** (too large and environment-specific).

## Generate the model artifacts

Run the training script once before starting the API:

```bash
python scripts/train_and_save_model.py
```

This creates:
- `model/xgboost_model.joblib` — trained XGBoost classifier
- `model/feature_names.json`   — list of expected feature names

## Model details

- Algorithm: XGBoost (XGBClassifier)
- Training data: Synthetic Telco churn dataset (generated for demo)
- Features: 13 customer attributes
- Target: Binary churn (0 = No Churn, 1 = Churn)
- Metric: ROC-AUC

For production use, replace with a model trained on real data
from the customer-churn-prediction project.
"""

# ─── scripts/train_and_save_model.py ───────────────────────────────────────
files["scripts/train_and_save_model.py"] = '''"""
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

    print(f"\\n✅ Model saved to {model_path}")
    print(f"✅ Feature names saved to {names_path}")
    print("\\nYou can now start the API:")
    print("  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
'''

# ─── tests/__init__.py ─────────────────────────────────────────────────────
files["tests/__init__.py"] = ""

# ─── tests/test_api.py ─────────────────────────────────────────────────────
files["tests/test_api.py"] = '''"""
test_api.py — Integration tests for all FastAPI endpoints.

Uses httpx.TestClient to test routes without running a real server.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_CUSTOMER = {
    "tenure":            2,
    "monthly_charges":   79.85,
    "total_charges":     159.70,
    "contract":          "Month-to-month",
    "internet_service":  "Fiber optic",
    "payment_method":    "Electronic check",
    "paperless_billing": True,
    "senior_citizen":    False,
    "partner":           False,
    "dependents":        False,
    "phone_service":     True,
    "online_security":   False,
    "tech_support":      False,
}

LOW_RISK_CUSTOMER = {
    "tenure":            60,
    "monthly_charges":   25.00,
    "total_charges":     1500.00,
    "contract":          "Two year",
    "internet_service":  "DSL",
    "payment_method":    "Bank transfer (automatic)",
    "paperless_billing": False,
    "senior_citizen":    False,
    "partner":           True,
    "dependents":        True,
    "phone_service":     True,
    "online_security":   True,
    "tech_support":      True,
}


class TestHealthEndpoint:

    def test_health_returns_200(self):
        """GET /health should always return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self):
        """Health response must contain status, model, version keys."""
        response = client.get("/health")
        data = response.json()
        assert "status"  in data
        assert "model"   in data
        assert "version" in data

    def test_health_status_ok(self):
        """Health status field must be 'ok'."""
        response = client.get("/health")
        assert response.json()["status"] == "ok"


class TestPredictEndpoint:

    def test_predict_returns_200(self):
        """POST /predict with valid input should return 200."""
        response = client.post("/predict", json=VALID_CUSTOMER)
        assert response.status_code == 200

    def test_predict_response_schema(self):
        """Prediction response must contain all required fields."""
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert "churn_probability" in data
        assert "churn_label"       in data
        assert "risk_tier"         in data
        assert "confidence"        in data

    def test_predict_probability_range(self):
        """Churn probability must be between 0 and 1."""
        response = client.post("/predict", json=VALID_CUSTOMER)
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_label_values(self):
        """Churn label must be either 'Churn' or 'No Churn'."""
        response = client.post("/predict", json=VALID_CUSTOMER)
        label = response.json()["churn_label"]
        assert label in {"Churn", "No Churn"}

    def test_predict_risk_tier_values(self):
        """Risk tier must be High, Medium, or Low."""
        response = client.post("/predict", json=VALID_CUSTOMER)
        tier = response.json()["risk_tier"]
        assert tier in {"High", "Medium", "Low"}

    def test_predict_invalid_contract_returns_422(self):
        """Invalid contract value should return 422 Unprocessable Entity."""
        bad_customer = {**VALID_CUSTOMER, "contract": "Weekly"}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_predict_negative_tenure_returns_422(self):
        """Negative tenure should fail Pydantic validation."""
        bad_customer = {**VALID_CUSTOMER, "tenure": -5}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_predict_missing_field_returns_422(self):
        """Missing required field should return 422."""
        bad_customer = {k: v for k, v in VALID_CUSTOMER.items()
                        if k != "monthly_charges"}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422


class TestBatchEndpoint:

    def test_batch_predict_returns_200(self):
        """POST /predict/batch with valid input should return 200."""
        response = client.post("/predict/batch",
                               json={"customers": [VALID_CUSTOMER,
                                                    LOW_RISK_CUSTOMER]})
        assert response.status_code == 200

    def test_batch_response_count(self):
        """Batch response total should match number of input customers."""
        customers = [VALID_CUSTOMER, LOW_RISK_CUSTOMER]
        response  = client.post("/predict/batch",
                                json={"customers": customers})
        data = response.json()
        assert data["total"] == len(customers)
        assert len(data["predictions"]) == len(customers)

    def test_batch_churn_rate_range(self):
        """Batch churn rate must be between 0 and 1."""
        response = client.post("/predict/batch",
                               json={"customers": [VALID_CUSTOMER,
                                                    LOW_RISK_CUSTOMER]})
        rate = response.json()["churn_rate"]
        assert 0.0 <= rate <= 1.0
'''

# ─── tests/test_preprocessing.py ───────────────────────────────────────────
files["tests/test_preprocessing.py"] = '''"""
test_preprocessing.py — Unit tests for feature preprocessing functions.
"""

import pytest
import numpy as np
from app.schemas       import CustomerFeatures
from app.preprocessing import (customer_to_features, get_risk_tier,
                                get_confidence)

SAMPLE = CustomerFeatures(
    tenure=12, monthly_charges=50.0, total_charges=600.0,
    contract="Month-to-month", internet_service="DSL",
    payment_method="Electronic check", paperless_billing=True,
    senior_citizen=False, partner=True, dependents=False,
    phone_service=True, online_security=False, tech_support=False,
)


class TestCustomerToFeatures:

    def test_output_is_numpy(self):
        result = customer_to_features(SAMPLE)
        assert isinstance(result, np.ndarray)

    def test_output_shape(self):
        result = customer_to_features(SAMPLE)
        assert result.shape == (1, 13)

    def test_no_nan(self):
        result = customer_to_features(SAMPLE)
        assert not np.isnan(result).any()

    def test_tenure_encoded_correctly(self):
        result = customer_to_features(SAMPLE)
        assert result[0][0] == 12.0


class TestRiskTier:

    def test_high_risk(self):
        assert get_risk_tier(0.85) == "High"

    def test_medium_risk(self):
        assert get_risk_tier(0.55) == "Medium"

    def test_low_risk(self):
        assert get_risk_tier(0.2) == "Low"

    def test_boundary_high(self):
        assert get_risk_tier(0.7) == "High"

    def test_boundary_medium(self):
        assert get_risk_tier(0.4) == "Medium"


class TestConfidence:

    def test_high_confidence_near_one(self):
        assert get_confidence(0.95) == "High"

    def test_high_confidence_near_zero(self):
        assert get_confidence(0.05) == "High"

    def test_low_confidence_near_boundary(self):
        assert get_confidence(0.5) == "Low"

    def test_medium_confidence(self):
        assert get_confidence(0.65) == "Medium"
'''

# ─── outputs/.gitkeep ──────────────────────────────────────────────────────
files["outputs/.gitkeep"] = ""
files["model/.gitkeep"]   = ""

# ─── Write all files ───────────────────────────────────────────────────────
def main():
    for filepath, content in files.items():
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  Created: {filepath}")

    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. python scripts/train_and_save_model.py")
    print("  3. uvicorn app.main:app --reload --port 8000")
    print("  4. Open http://localhost:8000/docs")
    print("  5. git init && git add . && git commit -m 'Add MLOps churn API'")
    print("  6. gh repo create mlops-churn-api --public --source=. --push")

if __name__ == "__main__":
    main()
