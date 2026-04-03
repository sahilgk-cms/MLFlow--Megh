from abc import ABC, abstractmethod


class BaseOptimizer(ABC):

    def __init__(self, trainer, param_space, direction, n_trials=None):
        self.trainer = trainer
        self.param_space = param_space
        self.direction = direction
        self.n_trails = n_trials

    @abstractmethod
    def optimize(self, X, y, **model_kwargs):
        pass


        