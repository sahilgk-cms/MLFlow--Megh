import optuna
import mlflow
from training.trainer import TimeSeriesTrainer
from optimizer.base import BaseOptimizer
from optuna.pruners import MedianPruner

class OptunaOptimizer(BaseOptimizer):
    def __init__(self, trainer: TimeSeriesTrainer, param_space, direction, n_trials: int):
        super().__init__(trainer, param_space, direction)
        self.n_trials = n_trials

    def optimize(self, X, y, **model_kwargs):
        metric = self.trainer.metric

        def objective(trial):
            params = self.param_space(trial)

            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)

                score = self.trainer.evaluate_params(X, y, params, **model_kwargs)

                mlflow.log_metric(f"cv_{metric.name}", score)

            return score

        study = optuna.create_study(direction=self.direction,
                                    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials)
        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
        }
