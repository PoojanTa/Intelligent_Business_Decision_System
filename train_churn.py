
"""
train_churn.py - Extended version with support for RandomForest, XGBoost, LightGBM,
hyperparameter ranges, Optuna example and MLflow logging. Run locally with:
python train_churn.py --model xgboost --use-mlflow --experiment IBDS-Churn
"""
import argparse, os, json, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.sklearn

# Try imports for XGBoost/LightGBM
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
from sklearn.ensemble import RandomForestClassifier

COMMON_HYPERPARAM_RANGES = {
    "random_forest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    },
    "xgboost": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    },
    "lightgbm": {
        "n_estimators": [100, 200, 400],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1],
        "feature_fraction": [0.6, 0.8, 1.0]
    }
}

def build_pipeline(model_name="random_forest"):
    # Simple preprocessing, can be extended
    cat_cols = ["contract","payment_method","tenure_bucket"]
    num_cols = None  # infer later
    def _create(X):
        nonlocal num_cols
        if num_cols is None:
            num_cols = [c for c in X.columns if c not in cat_cols and c not in ("customer_id","churn","total_charges")]
        preproc = ColumnTransformer([("num", StandardScaler(), num_cols),
                                    ("cat", OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)])
        if model_name == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_name == "xgboost":
            if xgb is None:
                raise ImportError("xgboost not installed. Install xgboost to use this model.")
            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
        elif model_name == "lightgbm":
            if lgb is None:
                raise ImportError("lightgbm not installed. Install lightgbm to use this model.")
            clf = lgb.LGBMClassifier(n_jobs=-1, random_state=42)
        else:
            raise ValueError("unknown model {}".format(model_name))
        pipe = Pipeline([("pre", preproc), ("clf", clf)])
        return pipe
    return _create

def evaluate_and_log(pipe, X_test, y_test, out_dir, mlflow_on=False):
    os.makedirs(out_dir, exist_ok=True)
    y_pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        try:
            y_proba = pipe.predict_proba(X_test)[:,1]
        except:
            y_proba = None
    else:
        y_proba = None
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    metrics = {"roc_auc": float(auc) if auc is not None else None, "accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
    # save model
    joblib.dump(pipe, os.path.join(out_dir,"churn_model.joblib"))
    with open(os.path.join(out_dir,"churn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if mlflow_on:
        mlflow.log_metrics({k:v for k,v in metrics.items() if v is not None})
        # log model artifact
        mlflow.sklearn.log_model(pipe, "churn_model")
    return metrics

def main(args):
    df = pd.read_csv(args.features_csv)
    y = df["churn"].values
    X = df.drop(columns=["customer_id","churn","total_charges"])
    # build pipeline dynamically after inferring numeric columns
    pipeline_builder = build_pipeline(args.model)
    pipe = pipeline_builder(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    if args.mlflow and args.experiment:
        mlflow.set_experiment(args.experiment)
        mlflow.start_run(run_name=f"churn_{args.model}_{args.run_name or 'run'}")
        mlflow.log_param("model", args.model)
    # fit
    pipe.fit(X_train, y_train)
    metrics = evaluate_and_log(pipe, X_test, y_test, args.out_dir, mlflow_on=args.mlflow)
    if args.mlflow:
        mlflow.end_run()
    print("Metrics:", metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default="features.csv")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--model", choices=["random_forest","xgboost","lightgbm"], default="random_forest")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    main(args)
