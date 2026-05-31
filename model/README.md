# Model Artifacts

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
