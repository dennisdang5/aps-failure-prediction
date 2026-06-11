import streamlit as st
import pandas as pd
import requests

st.title('APS Failure Prediction')

# Load sample data for the selector
X_test = pd.read_parquet('data/processed/X_test.parquet')

API_URL = 'http://localhost:8000/predict'
features = None

# Input method choice
input_method = st.radio('Choose input method:', ['Sample Truck', 'Upload CSV'])
if input_method == 'Sample Truck':
    truck_index = st.number_input('Truck index:', min_value=0, max_value=len(X_test)-1)
    features = X_test.iloc[int(truck_index)].to_dict()
else:
    expected_features = set(X_test.columns)
    uploaded = st.file_uploader('Upload sensor CSV', type='csv')
    if uploaded:
        df = pd.read_csv(uploaded)
        uploaded_features = set(df.columns)
        missing = expected_features - uploaded_features
        if missing:
            st.error(f"CSV is missing {len(missing)} required features")
        else:
            features = df.iloc[0].to_dict()

if st.button('Predict'):
    if features is None:
        st.warning('Please provide valid input first.')
    else:
        try:
            response = requests.post(API_URL, json={'features': features})
            response.raise_for_status()
            results = response.json()

            st.subheader('Result')
            st.write(f"Prediction: {results['predicted_class']}")
            col1, col2 = st.columns(2)
            col1.metric("Failure Probability", f"{results['failure_probability']:.2%}")
            col2.metric('Action', results['recommended_action'])
        except requests.exceptions.ConnectionError:
            st.error('Cannot connect to the API')
        except Exception as e:
            st.error(f'Prediction failed: {e}')