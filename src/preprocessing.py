import pandas as pd
from sklearn.impute import SimpleImputer

# Changes class labels to binary
def encode_labels(aps_data, target_column, positive_label, negative_label):
    aps_data[target_column] = aps_data[target_column].map({positive_label:1, negative_label:0})
    return aps_data

# Feature split
def split_features_target(aps_data, target_column):
    X = aps_data.drop(columns=[target_column])
    y = aps_data[target_column]
    return X, y

# Imputer for missing values
def impute_missing_values(X_train, X_test, strategy):
    imputer = SimpleImputer(strategy=strategy)
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    X_test_imputed = imputer.transform(X_test)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)
    return X_train_imputed, X_test_imputed
