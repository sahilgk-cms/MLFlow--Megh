import pandas as pd
import mlflow
from pipelines.prediction_builder import build_prediction_data
from utils.artifact_logger import log_parquet
from metrics.factory import MetricFactory

def run_evaluation_pipeline(
            X_test: pd.DataFrame,
            y_test: pd.Series,
            X_test_meta: pd.DataFrame,
            model,
            best_cv_score: float,
            predictions_path: str,
            ml_config: dict,
            pipeline_root_run_id: str,
    ):
    model_name = ml_config.get("model_name")
    run_type = ml_config.get("evaluation_run_type")

    with mlflow.start_run(run_name=f"{model_name}_{run_type}", nested=True):

        mlflow.set_tags({
            "model_name": model_name,
            "run_name": run_type,
            "pipeline_root_run_id": pipeline_root_run_id,
        })

        predictions = model.predict(X_test)
        eval_metrics = MetricFactory.get_eval_metrics(model_name)

        eval_metric_results = {}
        for metric in eval_metrics:
            value = metric.fn(y_test, predictions)
            mlflow.log_metric(f"test_{metric.name}", value)
            eval_metric_results[metric.name] = value

        # choose primary metric
        primary_metric = eval_metrics[0].name
        test_score = eval_metric_results[primary_metric]


        predictions_df = build_prediction_data(predictions,
                                               X_test_meta,
                                               best_cv_score,
                                               test_score,
                                               metric_name=primary_metric)

        log_parquet(
            df=predictions_df,
            filename=predictions_path,
            artifact_path="predictions",
        )

        return eval_metric_results
