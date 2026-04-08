from abc import ABC, abstractmethod

class BasePreprocessor(ABC):

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass

    @abstractmethod
    def get_cat_feature_indices(self):
        pass