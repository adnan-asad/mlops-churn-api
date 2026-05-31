# ⚙️ MLOps Churn Prediction API

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
source venv/bin/activate      # Windows: venv\Scripts\activate
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
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
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
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
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
