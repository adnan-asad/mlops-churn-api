"""
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
