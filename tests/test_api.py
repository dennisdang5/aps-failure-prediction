import pandas as pd
from fastapi.testclient import TestClient
from serve import app

client = TestClient(app)

def test_health_check():
    with TestClient(app) as client:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'

def test_predict():
    with TestClient(app) as client:
        X_test = pd.read_parquet('data/processed/X_test.parquet')
        features = X_test.iloc[0].to_dict()

        response = client.post('/predict', json={'features': features})

        assert response.status_code == 200
        result = response.json()

        # Check to make sure the response has the expected fields
        assert 'prediction' in result
        assert 'failure_probability' in result
        assert 'recommended_action' in result
        assert result['prediction'] in [0, 1]
        assert 0 <= result['failure_probability'] <= 1

def test_predict_missing_features():
    with TestClient(app) as client:
        # Send 2 features instead of 170
        response = client.post('/predict', json = {
            'features': {'aa_000': 1.0, 'ab_000': 2.0}
        })
        assert response.status_code == 422
        assert 'Missing' in response.json()['detail']