from models.catboost_model import CatBoostModel

class ModelFactory:
    REGISTRY = {
        "CatBoostRegressor": CatBoostModel
    }

    @classmethod
    def get_model(cls, model_name: str, **kwargs):
        if model_name not in cls.REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        model_cls = cls.REGISTRY[model_name]

        # Only pass extra kwargs (like cat features) — NOT params
        return model_cls, kwargs