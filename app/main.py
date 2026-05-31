"""
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
        "Production ML API for customer churn prediction.\n\n"
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
