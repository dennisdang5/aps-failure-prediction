import pandas as pd
import numpy as np
from preprocessing import encode_labels, split_features_target, impute_missing_values

def test_encode_labels():
    df = pd.DataFrame({'class': ['pos', 'neg', 'pos', 'neg']})
    result = encode_labels(df, 'class', 'pos', 'neg')
    assert list(result['class']) == [1, 0, 1, 0]

def test_split_features_target():
    df = pd.DataFrame({
        'features1': [1, 2, 3],
        'features2': [4, 5, 6],
        'class': [0, 1, 0]
    })

    X, y = split_features_target(df, 'class')

    assert 'class' not in X.columns
    assert list(y) == [0, 1, 0]
    assert X.shape == (3, 2)

def test_impute_missing_values():
    X_train = pd.DataFrame({
        'feature1': [1.0, 2.0, np.nan, 4.0],
        'feature2': [10.0, np.nan, 30.0, 40.0]
    })

    X_test = pd.DataFrame({
        'feature1': [np.nan, 5.0],
        'feature2': [50.0, np.nan]
    })

    X_train_imputed, X_test_imputed = impute_missing_values(X_train, X_test, 'median')

    assert X_train_imputed.isnull().sum().sum() == 0
    assert X_test_imputed.isnull().sum().sum() == 0
    assert X_train_imputed['feature1'].iloc[2] == 2.0