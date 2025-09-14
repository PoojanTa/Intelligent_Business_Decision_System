
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd
import traceback

app = FastAPI(title="IBDS Model Server - Churn & CLTV (local)", version="0.1")

class PredictRequest(BaseModel):
    customer_id: str
    features: dict  # pass precomputed features matching training feature set

@app.on_event("startup")
def load_models():
    global churn_pipe, cltv_model
    try:
        churn_pipe = joblib.load("artifacts/churn_model.joblib")
    except Exception as e:
        churn_pipe = None
    try:
        cltv_model = joblib.load("artifacts/cltv_model.joblib")
    except Exception as e:
        cltv_model = None

@app.get("/health")
def health():
    return {"status": "ok", "churn_model_loaded": churn_pipe is not None, "cltv_loaded": cltv_model is not None}

@app.post("/predict/churn")
def predict_churn(req: PredictRequest):
    try:
        if churn_pipe is None:
            return {"error": "churn model not loaded"}
        df = pd.DataFrame([req.features])
        proba = float(churn_pipe.predict_proba(df)[:,1][0])
        pred = int(churn_pipe.predict(df)[0])
        return {"customer_id": req.customer_id, "churn_proba": proba, "churn_pred": pred}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.post("/predict/cltv")
def predict_cltv(req: PredictRequest):
    try:
        if cltv_model is None:
            return {"error": "cltv model not loaded"}
        df = pd.DataFrame([req.features])
        pred = float(cltv_model.predict(df)[0])
        return {"customer_id": req.customer_id, "cltv_90d": pred}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
