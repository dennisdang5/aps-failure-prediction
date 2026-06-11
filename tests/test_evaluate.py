from evaluate import compute_cost, compute_misclassification_rate

def test_compute_cost():
    # Test inputs
    y_true = [0, 0, 1, 1, 1]
    y_pred = [1, 1, 0, 0, 0]
    cost_config = {'fp_cost':10, 'fn_cost': 500}

    result = compute_cost(y_true, y_pred, cost_config)

    # Expected: (2 FP x 10) + (3 FN x 500) = 1520
    assert result == 1520

def test_compute_cost_perfect_prediction():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    cost_config = {'fp_cost':10, 'fn_cost':500}

    result = compute_cost(y_true, y_pred, cost_config)

    assert result == 0

def test_compute_misclassification_rate():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    assert compute_misclassification_rate(y_true, y_pred) == 0.25