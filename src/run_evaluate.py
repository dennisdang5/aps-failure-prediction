import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from data_loader import load_config
from evaluate import compute_cost, compute_misclassification_rate

# Load config
config = load_config()

# Load test data
X_test = pd.read_parquet('data/processed/X_test.parquet')
y_test = pd.read_parquet('data/processed/y_test.parquet').squeeze()

# Load models
models_names = ['rf_imbalanced', 'rf_balanced', 'xgb', 'xgb_SMOTE']
models = {}
for name in models_names:
    models[name] = joblib.load(f'models/{name}.joblib')

# Metrics
results = []
for name, model in models.items():
    y_predictions = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_predictions).ravel()

    results.append({
        'Model': name,
        'Test AUC': roc_auc_score(y_test, y_probability),
        'FP': fp,
        'FN': fn,
        'Misclassification Rate': compute_misclassification_rate(y_test, y_predictions),
        'Cost': compute_cost(y_test, y_predictions, config['cost'])
    })

results_df = pd.DataFrame(results)
results_df.to_csv('results/model_comparison.csv', index=False)
print(results_df)