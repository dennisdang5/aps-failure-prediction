from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def train_random_forest_classifier(X_train, y_train, rf_config, class_weight=None):
    random_forest_classifier = RandomForestClassifier(
        n_estimators=rf_config['n_estimators'],
        oob_score=True,
        class_weight=class_weight,
        random_state=rf_config['random_state']
    )

    random_forest_classifier.fit(X_train, y_train)
    return random_forest_classifier

def train_xgb_classifier(X_train, y_train, xgb_config, cv_config):
    alphas = {'reg_alpha': xgb_config['alpha_grid']}
    xgb_classifier = XGBClassifier(
        booster=xgb_config['booster'],
        objective=xgb_config['objective'],
        random_state=xgb_config['random_state'],
        n_jobs=xgb_config['n_jobs']
    )

    grid_search_xgb_classifier = GridSearchCV(
        xgb_classifier,
        alphas,
        cv=cv_config['folds'],
        scoring=cv_config['scoring']
    )

    grid_search_xgb_classifier.fit(X_train, y_train)
    return grid_search_xgb_classifier

def train_SMOTE_xgb_classifier(X_train, y_train, xgb_config, cv_config, smote_config):
    alphas = {'xgb__reg_alpha': xgb_config['alpha_grid_smote']}
    smote_xgb_classifier_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=smote_config['random_state'])),
        ('xgb', XGBClassifier(booster=xgb_config['booster'],
                              objective=xgb_config['objective'],
                              random_state=xgb_config['random_state']))
    ])

    grid_search_xgb_SMOTE_model = GridSearchCV(
        smote_xgb_classifier_pipeline,
        alphas,
        cv=cv_config['folds'],
        scoring=cv_config['scoring']
    )

    grid_search_xgb_SMOTE_model.fit(X_train, y_train)
    return grid_search_xgb_SMOTE_model