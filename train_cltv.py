
"""
train_cltv.py - CLTV training with MLflow logging and hyperparameter hints.
Usage:
python train_cltv.py --mlflow --experiment IBDS-CLTV
"""
import argparse, os, json, joblib
import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow, mlflow.sklearn

HYPERPARAM_RANGES = {
    "hist_gb": {
        "max_iter": [100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_leaf_nodes": [31, 63, 127]
    }
}

def prepare_cltv(customers_csv="customers.csv", transactions_csv="transactions.csv", out="cltv_dataset.csv"):
    cust = pd.read_csv(customers_csv)
    tx = pd.read_csv(transactions_csv, parse_dates=["timestamp"])
    last_date = tx["timestamp"].max()
    cutoff = last_date - pd.Timedelta(days=90)
    future_spend = tx[tx["timestamp"] > cutoff].groupby("customer_id")["amount"].sum().rename("cltv_90d")
    base = cust.merge(future_spend, left_on="customer_id", right_index=True, how="left").fillna(0)
    base["cltv_90d"] = base["cltv_90d"].astype(float)
    base.to_csv(out, index=False)
    return out

def train_cltv(cltv_csv="cltv_dataset.csv", out_dir="artifacts", mlflow_on=False):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(cltv_csv)
    y = df["cltv_90d"].values
    X = df.drop(columns=["customer_id","cltv_90d","total_charges"])
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    metrics = {"rmse": float(rmse), "mae": float(mae)}
    joblib.dump(model, os.path.join(out_dir,"cltv_model.joblib"))
    with open(os.path.join(out_dir,"cltv_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if mlflow_on:
        mlflow.log_params({"model":"hist_gb"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "cltv_model")
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cltv-csv", default="cltv_dataset.csv")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()
    if args.mlflow and args.experiment:
        mlflow.set_experiment(args.experiment)
        mlflow.start_run(run_name="cltv_run")
    csv = prepare_cltv(args.cltv_csv.split(",")[0], "transactions.csv", "cltv_dataset.csv")
    metrics = train_cltv(csv, args.out_dir, mlflow_on=args.mlflow)
    if args.mlflow:
        mlflow.end_run()
    print("CLTV metrics:", metrics)
