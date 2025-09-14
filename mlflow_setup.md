
# MLflow Local Setup (brief)

To run MLflow server locally and use it from the scripts:
1. Create a directory to store artifacts, e.g. `./mlflow_artifacts`
2. Launch mlflow server:
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts --host 0.0.0.0 --port 5000
   ```
3. In your script runs, pass `--mlflow --experiment IBDS-Experiment` and set `MLFLOW_TRACKING_URI=http://localhost:5000` in the environment to ensure logs go to the local server.
4. Example run:
   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   python train_churn.py --model xgboost --mlflow --experiment IBDS-Churn
   ```
