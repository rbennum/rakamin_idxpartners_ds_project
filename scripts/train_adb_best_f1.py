import src.utils as utils
import src.models as models

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import category_encoders as ce

df_train = pd.read_parquet("input/df_train_02.parquet")
X = df_train.drop(columns=["loan_status"])
y = df_train["loan_status"]

# preprocessing pipeline
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = [
    "home_ownership",
    "verification_status",
    "term",
    "initial_list_status",
]
target_features = ["sub_grade", "purpose", "addr_state", "emp_length"]
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)
target_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "target_encoder",
            ce.TargetEncoder(handle_unknown="value", handle_missing="value"),
        ),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("tar", target_transformer, target_features),
    ],
    remainder="drop",
)

# base estimator for AdaBoost
custom_base_estimator = DecisionTreeClassifier(max_depth=1, random_state=29)

# main pipeline
pipeline = ImbPipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=29)),
        (
            "feature_selection",
            SelectFromModel(estimator=RandomForestClassifier(random_state=23)),
        ),
        (
            "classifier",
            AdaBoostClassifier(estimator=custom_base_estimator, random_state=29),
        ),
    ]
)
pipeline = ImbPipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=29)),
        (
            "feature_selection",
            SelectFromModel(estimator=RandomForestClassifier(random_state=23)),
        ),
        ("classifier", AdaBoostClassifier(random_state=29)),
    ]
)
param_grid = {
    "feature_selection__threshold": ["0.5*median", "0.75*median"],
    "smote__sampling_strategy": [0.5, 0.75],
    "classifier__n_estimators": [50, 100, 200],
    "classifier__learning_rate": [0.01, 0.1],
}

# customize cv and scorer
custom_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=29)
f1_scorer = make_scorer(f1_score, pos_label=1)
recall_scorer = make_scorer(recall_score, pos_label=1)
multiple_scorers = {
    "f1_score_bad_loan": f1_scorer,
    "recall_score_bad_loan": recall_scorer,
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    cv=custom_cv,
    scoring=multiple_scorers,
    refit="recall_score_bad_loan",
    verbose=2,
    n_jobs=-1,
)
print("Starting RandomizedSearchCV...")
random_search.fit(X, y)
print("Best parameters from grid search:")
print(random_search.best_params_)
timestamp = utils.get_time()
utils.save_model(random_search, filename=f"adb_random_{timestamp}.joblib")
