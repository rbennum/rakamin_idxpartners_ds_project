import src.utils as utils
import src.models as models

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

df_train = pd.read_csv('input/df_train.csv')
X = df_train.drop(columns=['loan_status'])
y = df_train['loan_status']

try:
    rf_random = utils.load_model('models/rf_randomized.joblib')
except Exception:
    preprocessor = models.create_preprocessing_pipeline(X)
    rf_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=29, class_weight='balanced', n_jobs=-1))
        ]
    )
    param_grid = {
        'classifier__n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        'classifier__max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__bootstrap': [True, False]
    }
    rf_random = RandomizedSearchCV(estimator=rf_pipeline, param_distributions=param_grid,
                                   n_iter=50, cv=3, random_state=29, n_jobs=-1,
                                   scoring='roc_auc')
    rf_random.fit(X, y)
    print('Best parameters from randomized search:')
    print(rf_random.best_params_)
    utils.save_model(rf_random, filename=f'rf_randomized_{utils.get_time()}.joblib')