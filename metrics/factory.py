from metrics.metrics import rmse
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: callable
    direction: str


class MetricFactory:

    REGISTRY = {
        "CatBoostRegressor":{
            "optimize": MetricSpec(
                name = "rmse",
                fn = rmse,
                direction = "minimize"
            ),
            "eval": [
                    MetricSpec(
                    name = "rmse",
                    fn = rmse,
                    direction = "minimize"
                    )
            ]
        }
    }


    @classmethod
    def get_optimize_metric(cls, model_name: str) -> MetricSpec:
        if model_name not in cls.REGISTRY:
            raise ValueError(f"No metric defined for model: {model_name}")
        return cls.REGISTRY[model_name]["optimize"]

    @classmethod
    def get_eval_metrics(cls, model_name: str) -> List[MetricSpec]:
        if model_name not in cls.REGISTRY:
            raise ValueError(f"No metric defined for model: {model_name}")
        return cls.REGISTRY[model_name]["eval"]
