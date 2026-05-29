from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt


def compute_misclassification_rate(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)

def compute_cost(y_true, y_pred, cost_config):
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = cm.ravel()
    total_cost = (cost_config['fp_cost'] * fp + cost_config['fn_cost'] * fn)
    return total_cost

def compute_cv_error(model, X_train, y_train, cv_config):
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_config['folds'], scoring='accuracy')
    return 1 - cv_scores.mean()

def plot_roc_curve(y_test, y_test_proba):
    RocCurveDisplay.from_predictions(y_test, y_test_proba)
    plt.show()