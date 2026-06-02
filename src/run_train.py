import joblib
import pandas as pd
from data_loader import load_config
from models import train_random_forest_classifier, train_xgb_classifier, train_SMOTE_xgb_classifier

# Load config
config = load_config()

# Read parquet files from data/processed/
X_train = pd.read_parquet('data/processed/X_train.parquet')
y_train = pd.read_parquet('data/processed/y_train.parquet').squeeze() # Transform single column df into series

# Train models
rf_imbalance_model = train_random_forest_classifier(X_train, y_train, config['random_forest'], class_weight=None)
rf_balance_model = train_random_forest_classifier(X_train, y_train, config['random_forest'], class_weight='balanced')
xgb_model = train_xgb_classifier(X_train, y_train, config['xgboost'], config['cross_validation'])
xgb_SMOTE_model = train_SMOTE_xgb_classifier(X_train, y_train, config['xgboost'], config['cross_validation'], config['smote'])

# Save model
joblib.dump(rf_imbalance_model, 'models/rf_imbalanced.joblib')
joblib.dump(rf_balance_model, 'models/rf_balanced.joblib')
joblib.dump(xgb_model, 'models/xgb.joblib')
joblib.dump(xgb_SMOTE_model, 'models/xgb_SMOTE.joblib')