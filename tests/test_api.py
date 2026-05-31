"""
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
