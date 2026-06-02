"""Preprocessing pipeline stage: load raw data, encode, split, impute, save"""
from data_loader import load_config, load_train_data, load_test_data
from preprocessing import encode_labels, split_features_target, impute_missing_values

# Load config
config = load_config()

# Load raw data
train_df = load_train_data(config['data'])
test_df = load_test_data(config['data'])

# Encode labels
train_df = encode_labels(
    train_df,
    config['data']['target_column'],
    config['data']['positive_label'],
    config['data']['negative_label']
)
test_df = encode_labels(
    test_df,
    config['data']['target_column'],
    config['data']['positive_label'],
    config['data']['negative_label']
)

# Split features and target
X_train, y_train = split_features_target(train_df, config['data']['target_column'])
X_test, y_test = split_features_target(test_df, config['data']['target_column'])

# Impute missing values
X_train_imputed, X_test_imputed = impute_missing_values(X_train, X_test, config['preprocessing']['imputation_strategy'])

# Save processed data to data/processed/
X_train_imputed.to_parquet('data/processed/X_train.parquet')
y_train.to_frame().to_parquet('data/processed/y_train.parquet')
X_test_imputed.to_parquet('data/processed/X_test.parquet')
y_test.to_frame().to_parquet('data/processed/y_test.parquet')

