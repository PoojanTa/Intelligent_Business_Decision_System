
# IBDS — Project Slide Deck (Detailed) 

## Slide 1 — Title
**Intelligent Business Decisioning System (IBDS)**
Purpose: End-to-end churn prediction, CLTV forecasting, and anomaly detection with production-ready MLOps.

## Slide 2 — Business Problem & Objectives
- Reduce churn, increase customer lifetime value (CLTV), detect anomalous/fraudulent transactions.
- KPIs: ROC-AUC (>0.7 realistic), lift on top-deciles, RMSE for CLTV, precision@k for anomalies, inference latency <200ms for online.

## Slide 3 — High-level Architecture
- Ingestion: Batch (CRM, billing) + Stream (transactions via Kafka)
- Feature Store: Delta/Parquet (Databricks)
- Modeling: XGBoost/LightGBM for tabular; LSTM autoencoder for sequence anomalies
- Serving: FastAPI / Kubernetes; MLflow for tracking; Grafana/Prometheus for monitoring; Power BI for stakeholder dashboards

## Slide 4 — Modeling Details (Churn)
- Candidate models: Logistic Regression (baseline), Random Forest, XGBoost, LightGBM
- Key features: RFM, avg_30d, avg_90d, tenure_bucket, contract, payment_method, monthly_norm, rfm_score
- Hyperparameter search ranges (example):
  - XGBoost: n_estimators [100,200,400], max_depth [3,6,10], learning_rate [0.01,0.05,0.1], subsample [0.6,0.8,1.0]
  - LightGBM: n_estimators [100,200,400], num_leaves [31,63,127], learning_rate [0.01,0.05,0.1]

## Slide 5 — CLTV Modeling
- Model: HistGradientBoostingRegressor (or XGBoost if preferred) with quantile regression for intervals
- Evaluation: RMSE, MAPE, business cost of over/underestimation

## Slide 6 — Anomaly Detection (Advanced)
- Hybrid setup: IsolationForest (transaction-level features) + LSTM autoencoder (sequence-level reconstruction errors)
- Aggregation rule: flag if either detector exceeds thresholds; human-in-loop for verification

## Slide 7 — Productionization & MLOps
- MLflow experiment tracking and artifact registry; DVC for data versioning
- CI/CD: tests + reproducible DVC stages; Docker & Kubernetes manifests provided
- Monitoring: Prometheus metrics + Grafana dashboards (prediction distribution, drift, latency)

## Slide 8 — Stakeholder Dashboards (Power BI)
- Churn overview: churn rate trend, top risk segments, per-campaign ROI simulator
- CLTV forecast: cohort CLTV, top customers by predicted CLTV, actionable list
- Anomaly alerts: timeline, severity, sample transactions for review

## Slide 9 — Next Steps & Runbook
- Integrate with company data sources, run A/B test for retention campaign, set up daily retrain pipeline, implement fairness checks

## Slide 10 — Contacts & Repo
- Repo: local IBDS package; run instructions in README.md
- Contact: add your name/contact

