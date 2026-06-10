from contextlib import asynccontextmanager
from fastapi import FastAPI
import mlflow
from pathlib import Path
from data_loader import load_config
from pydantic import BaseModel
import pandas as pd

# Config and model loading
config_path = Path(__file__).parent / 'config.yml'
config = load_config(str(config_path))
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

model = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = mlflow.sklearn.load_model('models:/aps-failure-predictor/latest')
    yield

# Pydantic model that validates incoming data
class PredictionRequest(BaseModel):
    features: dict[str, float]

# App creation
app = FastAPI(title='APS Failure Prediction API', lifespan=lifespan)

# Endpoints
@app.get('/health')
def health_check():
    return {'status': 'healthy', 'model_loaded': model is not None}

@app.post('/predict')
def predict(request: PredictionRequest):
    # Convert dictionary to single row df
    X = pd.DataFrame([request.features])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]

    return {
        'prediction': int(prediction),
        'failure_probability': float(probability),
        'predicted_class': 'failure' if prediction ==1 else 'no_failure'
    }