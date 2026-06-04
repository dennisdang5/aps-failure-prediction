import joblib
from data_loader import load_config, load_test_data
from evaluate import compute_cost, compute_cv_error, compute_misclassification_rate, plot_roc_curve

# Load config
config = load_config()

# Load test data
X_test = load_test_data(config['data'])

# Load models
models_names = ['rf_imbalanced', 'rf_balanced', 'xgb', 'xgb_SMOTE']
models = {}
for name in models_names:
    models[name] = joblib.load(f'models/{name}.joblib')
