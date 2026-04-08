from .preprocessor_v1 import TabularPreprocessorV1

class PreprocessorFactory:
    REGISTRY = {
        'tabular_v1': TabularPreprocessorV1,
    }

    @classmethod
    def create(cls, name:str, **kwargs):
        if name not in cls.REGISTRY:
            raise ValueError(f"Unknown preprocessor: {name}")
        return cls.REGISTRY[name](**kwargs)


