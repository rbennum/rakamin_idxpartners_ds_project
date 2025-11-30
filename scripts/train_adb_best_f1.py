import src.utils as utils
import src.models as models

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
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
numeric_features = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "credit_history_age",
]
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
            "classifier",
            AdaBoostClassifier(estimator=custom_base_estimator, random_state=29),
        ),
    ]
)
param_grid = {
    "smote__sampling_strategy": [0.75],
    "classifier__n_estimators": [100],
    "classifier__learning_rate": [0.1],
    "classifier__estimator__max_depth": [1],
}

# customize cv and scorer
custom_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=29)
custom_scorer = make_scorer(recall_score, pos_label=1)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    verbose=2,
    n_jobs=-1,
)
print("Starting RandomizedSearchCV...")
random_search.fit(X, y)
print("Best parameters from grid search:")
print(random_search.best_params_)
timestamp = utils.get_time()
utils.save_model(random_search, filename=f"adb_random_{timestamp}.joblib")
