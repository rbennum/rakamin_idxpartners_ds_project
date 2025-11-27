import src.utils as utils
import src.models as models

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv('input/df_train.csv')
X = df_train.drop(columns=['loan_status'])
y = df_train['loan_status']

try:
    rf_grid = utils.load_model('models/rf_grid.joblib')
except Exception:
    preprocessor = models.create_preprocessing_pipeline(X)
    rf_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=29, class_weight='balanced', n_jobs=-1))
        ]
    )
    param_grid_gs = {
        'classifier__n_estimators': [950, 1000, 1050],
        'classifier__max_depth': [65, 70, 75],
        'classifier__min_samples_split': [8, 10, 12],
        'classifier__min_samples_leaf': [4, 6],
        'classifier__bootstrap': [True]
    }
    rf_grid = GridSearchCV(estimator=rf_pipeline, param_grid=param_grid_gs, 
                           cv=3, n_jobs=-1, scoring='roc_auc')
    rf_grid.fit(X, y)
    print('Best parameters from grid search:')
    print(rf_grid.best_params_)
    utils.save_model(rf_grid, filename=f'rf_grid_{utils.get_time()}.joblib')
