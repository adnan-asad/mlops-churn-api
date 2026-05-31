"""
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
