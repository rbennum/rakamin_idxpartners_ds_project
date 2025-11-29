import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassNamePrefixFeaturesOutMixin,
)

from src import utils


class RiskBasedTopNEncoder(
    ClassNamePrefixFeaturesOutMixin, BaseEstimator, TransformerMixin
):
    def __init__(
        self,
        n_top_categories=10,
        min_cardinality_threshold=3,
        other_category_name="Other",
    ):
        self.n_top_categories = n_top_categories
        self.min_cardinality_threshold = min_cardinality_threshold
        self.other_category_name = other_category_name
        self.top_risk_categories_map_ = {}

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="target", index=X.index)

        temp_df = pd.concat([X, y], axis=1)
        target_col = y.name
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        self.feature_names_in_ = X.columns.tolist()

        for col in categorical_cols:
            if (
                col in temp_df.columns
                and X[col].nunique() >= self.min_cardinality_threshold
            ):
                risk = (
                    temp_df.groupby(col)[target_col].mean().sort_values(ascending=False)
                )
                top_n = risk.nlargest(self.n_top_categories).index.tolist()
                self.top_risk_categories_map_[col] = top_n
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X_transformed = pd.DataFrame(X)
        else:
            X_transformed = X.copy()
        for col, top_categories in self.top_risk_categories_map_.items():
            is_not_top = ~X_transformed[col].isin(top_categories)
            X_transformed.loc[is_not_top, col] = self.other_category_name
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.feature_names_in_)
        return np.array(input_features)


def create_preprocessing_pipeline(X_df):
    numeric_features = X_df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("topn", RiskBasedTopNEncoder(n_top_categories=3)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor
