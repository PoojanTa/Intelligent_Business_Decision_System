
# IBDS - Intelligent Business Decisioning System (Local version)

This local package contains scripts to run a compact, advanced pipeline demonstrating key components Quantiphi seeks in Machine Learning Engineers:
- Classification (churn) and Regression (CLTV)
- Feature engineering (RFM, rolling features)
- Anomaly detection (IsolationForest + autoencoder-like reconstructions)
- Reproducible modular scripts suitable for Databricks/Jupyter adaptation

## Quick start (local)
1. Create a venv and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the orchestrator (generates data and trains models):
   ```bash
   python quick_run.py
   ```
3. Artifacts (models, metrics, enriched transactions) will be in `./artifacts`.

## Files
- `data_gen.py` - synthetic customer and transactions generator
- `features.py` - feature engineering script that outputs `features.csv`
- `train_churn.py` - trains a churn classification model and saves metrics
- `train_cltv.py` - prepares CLTV dataset and trains a regressor
- `anomaly_detector.py` - trains anomaly detectors and saves artifacts
- `quick_run.py` - orchestrator to run the full pipeline locally
- `requirements.txt` - minimal dependencies for local run

## Notes & Next steps
- Swap synthetic data with company extracts (CRM, billing, events) by replacing CSVs.
- Add MLflow, DVC, and Optuna for experiment tracking & hyperparameter tuning.
- Replace MLP autoencoder with an LSTM autoencoder for sequence-sensitive anomaly detection (requires adding TensorFlow/PyTorch).

