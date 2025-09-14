
"""
anomaly_detector.py - Hybrid anomaly detection using IsolationForest + LSTM autoencoder.
This version builds simple sequences per customer (fixed-length) and trains a Keras LSTM autoencoder.
Logs artifacts and metrics to MLflow if requested.
Note: TensorFlow is required. This script organizes pipelines for local runs; for very large datasets use Databricks/PySpark.
"""
import os, json, joblib, argparse
import pandas as pd, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import mlflow, mlflow.sklearn, mlflow.tensorflow

# TensorFlow import is optional but required for LSTM autoencoder
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception as e:
    tf = None

def build_sequences(tx, seq_len=10):
    # tx sorted by timestamp. For each customer, build sliding windows of amounts of length seq_len
    tx = tx.sort_values(["customer_id","timestamp"])
    sequences = []
    meta = []
    for cid, g in tx.groupby("customer_id"):
        amounts = g["amount"].values
        for i in range(len(amounts) - seq_len + 1):
            seq = amounts[i:i+seq_len]
            sequences.append(seq)
            meta.append({"customer_id": cid, "start_idx": i, "is_anom": int(g["is_anom"].iloc[i+seq_len-1]) if "is_anom" in g.columns else 0})
    X = np.array(sequences)
    return X, pd.DataFrame(meta)

def lstm_autoencoder_model(seq_len, latent_dim=16):
    inputs = layers.Input(shape=(seq_len,1))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32, return_sequences=False)(x)
    encoded = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.RepeatVector(seq_len)(encoded)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    decoded = layers.TimeDistributed(layers.Dense(1))(x)
    model = models.Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_anomaly(transactions_csv="transactions.csv", out_dir="artifacts", seq_len=10, mlflow_on=False):
    os.makedirs(out_dir, exist_ok=True)
    tx = pd.read_csv(transactions_csv, parse_dates=["timestamp"])
    # basic features for IsolationForest input
    tx["day"] = tx["timestamp"].dt.dayofweek
    tx["hour"] = tx["timestamp"].dt.hour
    tx = tx.sort_values("timestamp")
    # compute simple rolling count in last 7 days
    tx["tx_count_7d"] = 0
    for cid, group in tx.groupby("customer_id"):
        times = group["timestamp"]
        counts = []
        for t in times:
            window_start = t - pd.Timedelta(days=7)
            c = ((times >= window_start) & (times < t)).sum()
            counts.append(c)
        tx.loc[group.index, "tx_count_7d"] = counts
    features = tx[["amount","day","hour","tx_count_7d"]].fillna(0)
    scaler = StandardScaler()
    X_feat = scaler.fit_transform(features)
    # Isolation Forest training
    iso = IsolationForest(n_estimators=200, contamination=0.005, random_state=42)
    iso.fit(X_feat)
    # Sequence-based LSTM autoencoder training
    X_seq, meta = build_sequences(tx, seq_len=seq_len)
    # normalize sequences per global scale
    X_seq = np.expand_dims(X_seq, -1)  # shape (n, seq_len, 1)
    if tf is None:
        raise ImportError("TensorFlow not installed. Install tensorflow to train the LSTM autoencoder.")
    model = lstm_autoencoder_model(seq_len, latent_dim=16)
    # train with early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_seq, X_seq, epochs=50, batch_size=64, callbacks=[es], verbose=1)
    # reconstruction error
    rec = model.predict(X_seq)
    rec_err = np.mean((rec - X_seq)**2, axis=(1,2))
    threshold = float(np.percentile(rec_err, 99))
    # use isolation forest predictions on per-transaction basis (reuse iso predictions)
    preds_iso = iso.predict(X_feat)  # -1 anomaly
    # align sequence-level rec_err back to transaction-level by marking the last transaction in the window
    tx_idx_flags = []
    for i, m in meta.iterrows():
        # map to index via customer grouping + start_idx; for simplicity mark based on original dataframe order
        tx_idx_flags.append(m["start_idx"])
    # create anomaly flag by sequence + iso
    anomalies = (rec_err > threshold)
    # create results DataFrame: for meta entries
    meta["rec_err"] = rec_err.tolist()
    meta["seq_anom"] = (meta["rec_err"] > threshold).astype(int)
    # evaluate against synthetic label if present
    p = None; r = None
    if "is_anom" in tx.columns:
        from sklearn.metrics import precision_score, recall_score
        # map meta seq_anom to underlying transaction label (meta.is_anom exists)
        p = precision_score(meta["is_anom"], meta["seq_anom"], zero_division=0)
        r = recall_score(meta["is_anom"], meta["seq_anom"], zero_division=0)
    # save artifacts
    joblib.dump(scaler, os.path.join(out_dir,"anomaly_scaler.joblib"))
    joblib.dump(iso, os.path.join(out_dir,"isolation_forest.joblib"))
    model.save(os.path.join(out_dir,"lstm_autoencoder_tf"))
    metrics = {"precision_on_synthetic_seq": p, "recall_on_synthetic_seq": r, "rec_threshold": threshold}
    with open(os.path.join(out_dir,"anomaly_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    meta.to_csv(os.path.join(out_dir,"anomaly_sequences_meta.csv"), index=False)
    if mlflow_on:
        mlflow.log_metrics({k:v for k,v in metrics.items() if v is not None})
        mlflow.sklearn.log_model(iso, "isolation_forest")
        mlflow.tensorflow.log_model(model, "lstm_autoencoder_tf")
    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions-csv", default="transactions.csv")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()
    if args.mlflow and args.experiment:
        mlflow.set_experiment(args.experiment)
        mlflow.start_run(run_name="anomaly_run")
    m = train_anomaly(args.transactions_csv, args.out_dir, seq_len=args.seq_len, mlflow_on=args.mlflow)
    if args.mlflow:
        mlflow.end_run()
    print("Anomaly metrics", m)
