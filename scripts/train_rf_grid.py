import src.utils as utils
import src.models as models

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df_train = pd.read_parquet("input/df_train_02.parquet")
X = df_train.drop(columns=["loan_status"])
y = df_train["loan_status"]

# preprocessing pipeline
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("topn", models.RiskBasedTopNEncoder(n_top_categories=3)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# main pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "feature_selection",
            SelectFromModel(estimator=RandomForestClassifier(random_state=23)),
        ),
        ("classifier", RandomForestClassifier(random_state=29)),
    ]
)
param_grid = {
    "feature_selection__threshold": ["mean", "median", "1.25*mean"],
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 10, 20],
}
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    verbose=2,
    n_jobs=-1,
)
print("Starting GridSearchCV...")
grid_search.fit(X, y)
print("Best parameters from grid search:")
print(grid_search.best_params_)
timestamp = utils.get_time()
utils.save_model(grid_search, filename=f"rf_grid_{timestamp}.joblib")
