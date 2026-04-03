from itertools import product
from optimizer.base import BaseOptimizer
import mlflow


class GridSearchOptimizer(BaseOptimizer):

    def optimize(self, X, y, **model_kwargs):

        keys = list(self.param_space.keys())
        values = list(self.param_space.values())

        best_score = None
        best_params = None
        trial_number = 0

        for combination in product(*values):

            params = dict(zip(keys, combination))

            with mlflow.start_run(run_name=f"grid_trial_{trial_number}", nested=True):

                mlflow.log_params(params)

                score = self.trainer.evaluate_params(
                    X, y, params, **model_kwargs
                )

                mlflow.log_metric(
                    f"cv_{self.trainer.metric.name}",
                    score
                )

            if best_score is None:
                best_score = score
                best_params = params

            else:

                if self.direction == "minimize":
                    if score < best_score:
                        best_score = score
                        best_params = params

                else:
                    if score > best_score:
                        best_score = score
                        best_params = params

            trial_number += 1

        return {
            "best_params": best_params,
            "best_score": best_score
        }




    