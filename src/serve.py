from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import HTTPException
from pathlib import Path
from data_loader import load_config
from pydantic import BaseModel
import pandas as pd
import joblib
import json

# Config and model loading
config_path = Path(__file__).parent / 'config.yml'
config = load_config(str(config_path))

model = None
expected_features = None
prediction_count = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, expected_features
    model_path = Path(__file__).parent.parent / config['serve']['model_path']
    model = joblib.load(model_path)

    # Load expected features names from the training data
    features_path = Path(__file__).parent / 'features_names.json'
    with open(features_path) as f:
        expected_features = json.load(f)
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
    global prediction_count

    # Validate that all expected features are present from the user
    missing = set(expected_features) - set(request.features.keys())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f'Missing {len(missing)} required features'
        )

    prediction_count += 1
    # Convert dictionary to single row df
    X = pd.DataFrame([request.features])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]

    # To break even probability x $500 > $10 thus probability > 0.02
    threshold = config['cost']['decision_threshold']
    if probability >= threshold:
        action = 'inspect'
    else:
        action = 'no_action'

    return {
        'prediction': int(prediction),
        'failure_probability': float(probability),
        'predicted_class': 'failure' if prediction == 1 else 'no_failure',
        'recommended_action': action
    }

@app.get('/metrics')
def metrics():
    return {
        'model_loaded': model is not None,
        'model_name': 'aps-failure-predictor',
        'decision_threshold': config['cost']['decision_threshold'],
        'prediction_count': prediction_count
    }