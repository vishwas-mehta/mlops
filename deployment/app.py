from fastapi import FastAPI, Request, Response, status, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import time
import os

app = FastAPI()

# Global state
app.state = {"is_alive": True, "is_ready": False}

# Load model
try:
    model = joblib.load('artifacts/heart_disease_model.joblib')
    app.state["is_ready"] = True
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Schema
class HeartDataSchema(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API - OPPE2"}

@app.get("/live_check")
async def liveness_probe():
    if app.state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check")
async def readiness_probe():
    if not app.state["is_ready"] or model is None:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return {"status": "ready"}

@app.post("/predict")
async def predict(data: HeartDataSchema):
    start_time = time.time()
    
    if not app.state["is_ready"] or model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Predict
        output = model.predict(input_data)[0]
        latency = round((time.time() - start_time) * 1000, 2)
        
        logger.info(f"Prediction completed, latency_ms: {latency}")
        
        return {
            "predicted_class": int(output),
            "latency_ms": latency
        }
    
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
