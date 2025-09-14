
import subprocess, os, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
os.chdir(ROOT)

print("1) Generating data...")
subprocess.run([sys.executable, "data_gen.py"], check=True)

print("2) Creating features...")
subprocess.run([sys.executable, "-c", "from features import create_features; create_features('customers.csv','transactions.csv','features.csv')"], check=True)

print("3) Preparing CLTV dataset...")
subprocess.run([sys.executable, "-c", "from train_cltv import prepare_cltv; prepare_cltv('customers.csv','transactions.csv','cltv_dataset.csv')"], check=True)

print("4) Training churn model...")
subprocess.run([sys.executable, "train_churn.py"], check=True)

print("5) Training CLTV model...")
subprocess.run([sys.executable, "train_cltv.py"], check=True)

print("6) Training anomaly detector...")
subprocess.run([sys.executable, "anomaly_detector.py"], check=True)

print('All steps complete. Artifacts in ./artifacts')
with open("artifacts/summary.json","w") as f:
    import json, os
    m = {}
    for fn in os.listdir("artifacts"):
        if fn.endswith(".json"):
            with open(os.path.join("artifacts",fn)) as fh:
                try:
                    m[fn] = json.load(fh)
                except:
                    m[fn] = "err"
    json.dump(m, f, indent=2)
print('Summary written to artifacts/summary.json')
