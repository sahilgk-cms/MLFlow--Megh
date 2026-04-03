from catboost import CatBoostRegressor, Pool
from .base import BaseModel
import mlflow
import pandas as pd


class CatBoostModel(BaseModel):
    def __init__(self, params: dict, cat_feature_indices: list, ml_config: dict):
        self.params = params
        self.cat_feature_indices = cat_feature_indices
        self.ml_config = ml_config
        self.fixed_params = {
            'loss_function': 'Poisson',
            'random_state': 42,
            'verbose': False,
            'task_type': "GPU" if self.ml_config.get("use_gpu", False) else "CPU",
        }

    @classmethod
    def from_params(cls, params: dict, **kwargs):
        return cls(params = params,
                   cat_feature_indices=kwargs.get('cat_feature_indices'),
                   ml_config =  kwargs.get("ml_config"))


    def fit(self, X, y):
        train_pool = Pool(X, y, cat_features=self.cat_feature_indices)
        self.model = CatBoostRegressor(**self.params, **self.fixed_params)
        self.model.fit(train_pool)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model

    def log_to_mlflow(self):
        mlflow.catboost.log_model(
            self.model,
            artifact_path="model"
        )
    
    def has_feature_importance(self):
        return True

    def get_feature_importance(self, feature_names):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        return importance_df



