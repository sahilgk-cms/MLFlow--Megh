from optimizer.optuna_optimizer import OptunaOptimizer
from optimizer.grid_optimizer import GridSearchOptimizer
from optimizer.random_optimizer import RandomSearchOptimizer

class OptimizerFactory:
    REGISTRY = {
        "optuna": OptunaOptimizer,
        "grid": GridSearchOptimizer,
        "random": RandomSearchOptimizer,
    }
     
    @classmethod
    def create(cls, optimizer_type):
        if optimizer_type not in cls.REGISTRY:
            raise ValueError(f"unknown optimizer type: {optimizer_type}")
        
        return cls.REGISTRY[optimizer_type]
        
