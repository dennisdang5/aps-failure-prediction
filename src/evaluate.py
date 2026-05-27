from sklearn.metrics import accuracy_score

def compute_misclassification_rate(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)

