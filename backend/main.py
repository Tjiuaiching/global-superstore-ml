import pandas as pd
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Global Superstore ML API",
    description="API for profit prediction and customer segmentation",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

regression_pipeline = joblib.load(
    MODEL_DIR / "regression_pipeline.pkl"
)

clustering_pipeline = joblib.load(
    MODEL_DIR / "clustering_pipeline.pkl"
)

cluster_labels = joblib.load(
    MODEL_DIR / "cluster_labels.pkl"
)

class RegressionInput(BaseModel):
    discount: float
    quantity: int
    sales: float
    shipping_cost: float
    category: str
    segment: str
    ship_mode: str
    order_priority: str
    market: str

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "regression_model": "loaded",
        "clustering_model": "loaded"
    }

@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    input_df = pd.DataFrame([data.model_dump()])

    prediction = regression_pipeline.predict(input_df)[0]

    return {
        "predicted_profit": float(prediction)
    }

class ClusteringInput(BaseModel):
    total_spending: float
    order_count: int
    average_order_value: float
    average_discount: float

@app.post("/predict/clustering")
def predict_clustering(data: ClusteringInput):
    input_df = pd.DataFrame([data.model_dump()])

    cluster_number = clustering_pipeline.predict(input_df)[0]
    segment_label = cluster_labels[cluster_number]

    return {
        "cluster": int(cluster_number),
        "segment": segment_label
    }