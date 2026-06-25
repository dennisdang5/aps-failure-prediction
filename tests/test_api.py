import pandas as pd
from fastapi.testclient import TestClient
from serve import app

client = TestClient(app)

def test_health_check():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'

def test_predict():
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