from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from .base import BasePreprocessor


class TabularPreprocessorV1(BasePreprocessor):
    def __init__(self):
        self.preprocessor = None
        self.num_cols = None
        self.cat_cols = None
        self.feature_names = None
        self.cat_feature_indices = None


    def fit(self, X: pd.DataFrame):
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), self.num_cols),
                ('cat', 'passthrough', self.cat_cols)
            ],
            remainder='drop'
        )

        self.preprocessor.fit(X)
        self.feature_names = self.num_cols + self.cat_cols
        self.cat_feature_indices = list(range(len(self.num_cols), len(self.num_cols) + len(self.cat_cols)))
        return self

    def transform(self, X: pd.DataFrame):
        if self.preprocessor is None:
            raise RuntimeError('Preprocessor not fitted')
        return self.preprocessor.transform(X)

    def get_feature_names(self):
        return  self.feature_names

    def get_cat_feature_indices(self):
        return self.cat_feature_indices

    