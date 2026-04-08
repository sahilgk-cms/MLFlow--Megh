from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict, **kwargs):
        """Create model from hyperparameters"""
        pass

    def has_feature_importance(self):
        return False

    def get_feature_importance(self, feature_names):
        raise NotImplementedError
