import mlflow
from mlflow.tracking import MlflowClient
from config.env import MLFLOW_URI
import pandas as pd
from pathlib import Path
from typing import Dict
import time
from utils.helpers import safe_tag_value

def initiate_client(mlflow_uri: str):
    client = MlflowClient(tracking_uri=mlflow_uri)
    return client


def start_mlflow_experiment(mlflow_uri: str, experiment_name: str, artifact_location: str=None):
    mlflow.set_tracking_uri(mlflow_uri)

    existing_exp = mlflow.get_experiment_by_name(experiment_name)

    if existing_exp:
        experiment_id = existing_exp.experiment_id
    else:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location
        )

    mlflow.set_experiment(experiment_name)

    return mlflow.get_experiment(experiment_id)


def register_model_with_data_tags(client,
                                 training_run_id: str,
                                  experiment_name: str,
                                  features_config: dict,
                                  data_config: dict,
                                  ml_config: dict,
                                  train_data_hash: str,
                                  test_data_hash: str,
                                  pipeline_root_run_id: str,
                                  eval_metric_results: dict) -> mlflow.entities.model_registry.model_version.ModelVersion:

    model_name = ml_config.get("model_name")
    registered_model_name = f"{experiment_name}_{model_name}"
    preprocessor_name = ml_config.get("preprocessor_name")
    optimizer_type = ml_config.get("optimizer_type")

    model_uri = f"runs:/{training_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name= registered_model_name)

    # WAIT until model version is READY
    for _ in range(10):
        mv_status = client.get_model_version(
            name=registered_model_name,
            version=mv.version
        )
        
        if mv_status.status == "READY":
            break
        
        time.sleep(1)

    for key, value in features_config.items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=mv.version,
            key=key,
            value=safe_tag_value(value)
        )

    for key, value in data_config.items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=mv.version,
            key=key,
            value=safe_tag_value(value)
        )

    client.set_model_version_tag(
        name=registered_model_name,
        version=mv.version,
        key="train_data_hash",
        value=train_data_hash
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=mv.version,
        key="test_data_hash",
        value=test_data_hash
    )

    client.set_model_version_tag(
        name= registered_model_name,
        version=mv.version,
        key="pipeline_root_run_id",
        value=pipeline_root_run_id
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=mv.version,
        key="preprocessor_name",
        value=preprocessor_name
    )

    client.set_model_version_tag(
        name= registered_model_name,
        version=mv.version,
        key="optimizer_type",
        value=optimizer_type
    )

    for metric_name, value in eval_metric_results.items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=mv.version,
            key=f"test_{metric_name}",
            value=str(value)
        )
    return mv

def load_model_from_registry(
    registered_model_name: str,
    stage: str | None = None,
    version: int | None = None) -> mlflow.pyfunc.PyFuncModel:

    if stage:
        model_uri = f"models:/{registered_model_name}/{stage}"
    elif version:
        model_uri = f"models:/{registered_model_name}/{version}"
    else:
        raise ValueError("Provide either stage or version")

    model = mlflow.pyfunc.load_model(model_uri)
    return model

def get_training_context(client, registered_model_name: str, version: int) -> dict:
    mv = client.get_model_version(registered_model_name, version)
    run = client.get_run(mv.run_id)

    test_metrics = {
        k:v 
        for k, v in mv.tags.items()
        if k.startswith("test_") and k != "test_data_hash"
    }

    context = {
        "training_run_id": mv.run_id,
        "pipeline_root_run_id": mv.tags.get("pipeline_root_run_id"),
        "experiment_id": run.info.experiment_id,
        "train_data_hash": mv.tags.get("train_data_hash"),
        "test_data_hash": mv.tags.get("test_data_hash"),
        "params": run.data.params,
        "metrics": run.data.metrics,
        "test_metrics": test_metrics,
        "tags": run.data.tags,

    }
    return context

def load_train_test_data(client, registered_model_name: str, version: int) -> dict:
    training_context = get_training_context(client, registered_model_name, version)
    pipeline_root_run_id = training_context["pipeline_root_run_id"]

    local_path = mlflow.artifacts.download_artifacts(
        run_id=pipeline_root_run_id,
        artifact_path="data"
    )

    return {
        file.name: pd.read_parquet(file)
        for file in Path(local_path).glob("*.parquet")
    }

def load_predictions(client, registered_model_name: str, version: int) -> Dict[str, pd.DataFrame]:
    training_context = get_training_context(client, registered_model_name, version)
    pipeline_root_run_id = training_context["pipeline_root_run_id"]
    experiment_id = training_context["experiment_id"]

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.pipeline_root_run_id = '{pipeline_root_run_id}' "
                      f"and tags.model_name = '{registered_model_name}' "
                      f"and tags.run_name = 'evaluation'",
        order_by=["start_time DESC"],
        max_results=1
         )
    run_id = runs[0].info.run_id
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="predictions"
        )
    return {
        file.name: pd.read_parquet(file)
        for file in Path(local_path).glob("*.parquet")
    }
