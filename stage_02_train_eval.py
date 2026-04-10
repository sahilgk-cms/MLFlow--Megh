import mlflow
import pickle
from datetime import datetime
import os

from preprocessing.factory import PreprocessorFactory
from training.trainer import TimeSeriesTrainer
from pipelines.train_pipeline import run_training_pipeline
from pipelines.evaluation_pipeline import run_evaluation_pipeline
from search_space.search_space import get_search_space

from utils.mlflow_helpers import start_mlflow_experiment, register_model_with_data_tags, initiate_client, log_git_to_mlflow, log_dvc_info
from utils.explainability import log_shap_summary
from utils.artifact_logger import log_parquet
from utils.helpers import load_yaml_config
from utils.hardware import detect_gpu

from config.env import MLFLOW_URI
from config.filepaths import DATA_ARTIFACT, TRAIN_PATH, TEST_PATH, FEATURE_IMPORTANCE_PATH, PREDICTIONS_PATH, SHAP_SUMMARY_PATH, SHAP_VALUES_PATH, RUN_ARTIFACT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ml_config")
parser.add_argument("--search_space")
parser.add_argument("--data_config")
parser.add_argument("--database_config")
parser.add_argument("--feature_config")

args = parser.parse_args()

DATA_CONFIG = load_yaml_config(args.data_config)
DATABASE_CONFIG = load_yaml_config(args.database_config)
FEATURE_CONFIG = load_yaml_config(args.feature_config)
ML_CONFIG = load_yaml_config(args.ml_config)
ML_CONFIG["use_gpu"] = detect_gpu()["available"]

def main():

    # load artifacts
    with open(DATA_ARTIFACT, "rb") as f:
        output = pickle.load(f)

    # ⚠️ run_id is optional now (do NOT fail pipeline if missing)
    parent_run_id = None
    run_id_path = RUN_ARTIFACT

    if os.path.exists(run_id_path):
        with open(run_id_path) as f:
            parent_run_id = f.read().strip()

    X_train = output["features"]["X_train"]
    y_train = output["features"]["y_train"]
    X_test = output["features"]["X_test"]
    y_test = output["features"]["y_test"]

    # 🔹 preprocessing
    pre = PreprocessorFactory.create(ML_CONFIG.get("preprocessor_name"))
    pre.fit(X_train)

    X_train_p = pre.transform(X_train)
    X_test_p = pre.transform(X_test)

    feature_names = pre.get_feature_names()
    cat_feature_indices = pre.get_cat_feature_indices()

    #  MLflow attach
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = initiate_client(MLFLOW_URI)

    state = DATABASE_CONFIG.get('state').replace(" ", "_")
    disease = DATABASE_CONFIG.get('disease').replace(" ", "_")

    experiment_name = f"{state}_{disease}"
    experiment = start_mlflow_experiment(mlflow_uri=MLFLOW_URI,
                                         experiment_name=experiment_name,
                                         )
    
    today_date = datetime.now().strftime("%Y/%m/%d")
    with mlflow.start_run(run_name = f"{experiment_name}_pipeline_root_{today_date}") as root_run:
        log_git_to_mlflow()
        log_dvc_info()

        log_parquet(df = output["data"]["train_df"], filename=TRAIN_PATH, artifact_path="data")
        log_parquet(df=output["data"]["test_df"], filename=TEST_PATH, artifact_path="data")


        pipeline_root_run_id = root_run.info.run_id
        # training

        search_space = get_search_space(
            args.search_space,
            model_name=ML_CONFIG.get("model_name"),
            optimizer_type=ML_CONFIG.get("optimizer_type")
        )

        final_model, best_cv_score, best_params, training_run_id = run_training_pipeline(
            X_train=X_train_p,
            y_train=y_train,
            ml_config=ML_CONFIG,
            trainer_cls=TimeSeriesTrainer,
            search_space=search_space,
            pipeline_root_run_id=pipeline_root_run_id,
            cat_feature_indices=cat_feature_indices
        )

        if final_model.has_feature_importance():
            importance_df = final_model.get_feature_importance(feature_names=feature_names)
            #importance_df = importance_df[importance_df["feature"] != "case_count_next_week"]
            log_parquet(df=importance_df, filename=FEATURE_IMPORTANCE_PATH,
                         artifact_path="feature_importance")

        # evaluation
        metric_results = run_evaluation_pipeline(
            X_test=X_test_p,
            y_test=y_test,
            X_test_meta=output["test_meta"],
            model=final_model,
            best_cv_score=best_cv_score,
            predictions_path=PREDICTIONS_PATH,
            ml_config=ML_CONFIG,
            pipeline_root_run_id=parent_run_id
        )

        register_model_with_data_tags(
            client = client,
            experiment_name=experiment_name,
            training_run_id=training_run_id,
            features_config=FEATURE_CONFIG,
            data_config=DATA_CONFIG,
            ml_config=ML_CONFIG,
            train_data_hash=output["hash"]["train_data_hash"],
            test_data_hash=output["hash"]["test_data_hash"],
            pipeline_root_run_id=pipeline_root_run_id,
            eval_metric_results=metric_results
        )


        # explainability
        shap_summary_path, shap_df = log_shap_summary(
            model_wrapper=final_model,
            X_sample=X_test_p,
            feature_names=feature_names,
            shap_summary_path=SHAP_SUMMARY_PATH
        )

        log_parquet(shap_df, SHAP_VALUES_PATH, artifact_path="explainability")
        mlflow.log_artifact(shap_summary_path, artifact_path="explainability")

if __name__ == "__main__":
    main()