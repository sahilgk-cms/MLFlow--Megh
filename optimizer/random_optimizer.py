import random
import mlflow
from optimizer.base import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):

    def __init__(self, trainer, param_space, direction, n_trials=20):
        super().__init__(trainer, param_space, direction)
        self.n_trials = n_trials

    def optimize(self, X, y, **model_kwargs):

        best_score = None
        best_params = None

        keys = list(self.param_space.keys())

        for trial in range(self.n_trials):

            # sample parameters randomly
            params = {
                k: random.choice(self.param_space[k])
                for k in keys
            }

            with mlflow.start_run(run_name=f"random_trial_{trial}", nested=True):

                mlflow.log_params(params)

                score = self.trainer.evaluate_params(
                    X,
                    y,
                    params,
                    **model_kwargs
                )

                mlflow.log_metric(
                    f"cv_{self.trainer.metric.name}",
                    score
                )

            # update best params
            if best_score is None:
                best_score = score
                best_params = params

            else:

                if self.direction == "minimize":
                    if score < best_score:
                        best_score = score
                        best_params = params

                else:  # maximize
                    if score > best_score:
                        best_score = score
                        best_params = params

        return {
            "best_params": best_params,
            "best_score": best_score
        }