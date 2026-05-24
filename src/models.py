from sklearn.ensemble import RandomForestClassifier

def train_random_forest_classifier(X_train, y_train, class_weight=None, n_estimators=100, oob_score=True, random_state=42):
    random_forest_classifier = RandomForestClassifier(n_estimators=n_estimators, oob_score=oob_score, class_weight=class_weight, random_state=random_state)
    random_forest_classifier.fit(X_train, y_train)
    return random_forest_classifier