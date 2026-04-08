import numpy as np

class TimeSeriesTrainer:
    def __init__(self, model_cls, cv, metric):
        """
        metric: MetricSpec (name, fn, direction)
        """
        self.model_cls = model_cls
        self.cv = cv
        self.metric = metric

    def evaluate_params(self, X, y, params, **model_kwargs) -> float:
        scores = []

        for train_idx, val_idx in self.cv.split(X):
            model = self.model_cls.from_params(params, **model_kwargs)
            model.fit(X[train_idx], y.iloc[train_idx])

            preds = model.predict(X[val_idx])
            score = self.metric.fn(y.iloc[val_idx], preds)
            scores.append(score)

        return float(np.mean(scores))

    def train_final(self, X, y, best_params, **model_kwargs):
        model = self.model_cls.from_params(best_params, **model_kwargs)
        model.fit(X, y)
        return model
