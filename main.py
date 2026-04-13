from pipelines.features_builder import build_features
from pipelines.data_builder import build_data
from db.engine import get_engine
from datetime import datetime
from preprocessing.factory import PreprocessorFactory
from utils.mlflow_helpers import start_mlflow_experiment, log_git_to_mlflow, log_dvc_info, safe_end_run
from utils.artifact_logger import log_parquet, log_config
from utils.mlflow_helpers import register_model_with_data_tags, initiate_client
from utils.explainability import log_shap_summary
from training.trainer import TimeSeriesTrainer
from pipelines.train_pipeline import run_training_pipeline
from pipelines.evaluation_pipeline import run_evaluation_pipeline
from search_space.search_space import get_search_space
import mlflow
from config.env import DB_NAME, DB_PASSWORD, DB_HOST, DB_PORT, DB_USER
from config.env import MLFLOW_URI
from config.filepaths import FEATURES_ARTIFACT, VILLAGE_EMBEDDINGS_PATH, PREDICTIONS_PATH, SHAP_SUMMARY_PATH, SHAP_VALUES_PATH, FEATURE_IMPORTANCE_PATH, TRAIN_PATH, TEST_PATH
from log.logger import get_logger
from utils.helpers import safe_tag_value, load_yaml_config
from utils.hardware import detect_gpu
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_config")
parser.add_argument("--database_config")
parser.add_argument("--feature_config")
parser.add_argument("--ml_config")
parser.add_argument("--search_space")

args = parser.parse_args()

DATA_CONFIG = load_yaml_config(args.data_config)
DATABASE_CONFIG = load_yaml_config(args.database_config)
FEATURE_CONFIG = load_yaml_config(args.feature_config)
ML_CONFIG = load_yaml_config(args.ml_config)

ML_CONFIG["use_gpu"] = detect_gpu()["available"]

for key, ranges in FEATURE_CONFIG["bucket_defs"].items():
    FEATURE_CONFIG["bucket_defs"][key] = [tuple(r) for r in ranges]

logger = get_logger(__name__)

def main():
   
    engine = get_engine(db_user=DB_USER, db_password=DB_PASSWORD, db_host=DB_HOST,
                        db_port=DB_PORT, db_name=DB_NAME)
    
    logger.info("Building features..")
    statewise_final = build_features(engine=engine, database_config=DATABASE_CONFIG,
                                    feature_config=FEATURE_CONFIG, village_embeddings_path=VILLAGE_EMBEDDINGS_PATH)
    log_parquet(df=statewise_final, filename=FEATURES_ARTIFACT,
                         artifact_path="features")

    logger.info("Building data...")
    output = build_data(df=statewise_final, data_config=DATA_CONFIG)
    

    X_train = output["features"]["X_train"]
    y_train = output["features"]["y_train"]
    X_test = output["features"]["X_test"]
    y_test = output["features"]["y_test"]

    logger.info("Preprocessing data...")
    pre = PreprocessorFactory.create(ML_CONFIG.get("preprocessor_name"))
    pre.fit(X_train)
    X_train_preprocessed = pre.transform(X_train)
    X_test_preprocessed = pre.transform(X_test)

    feature_names = pre.get_feature_names()
    cat_feature_indices = pre.get_cat_feature_indices()

    logger.info("Starting MLFlow...")
    state = (DATABASE_CONFIG.get('state')).replace(" ", "_")
    disease = (DATABASE_CONFIG.get('disease')).replace(" ", "_")

    experiment_name = f"{state}_{disease}"
    experiment = start_mlflow_experiment(mlflow_uri=MLFLOW_URI,
                                         experiment_name=experiment_name,
                                         )
    today_date = datetime.now().strftime("%Y/%m/%d")


    mlflow.set_tracking_uri(MLFLOW_URI)
    client = initiate_client(MLFLOW_URI)

    safe_end_run()
    with mlflow.start_run(run_name = f"{experiment_name}_pipeline_root_{today_date}") as pipeline_root:
        log_git_to_mlflow()
        log_dvc_info()
       
        pipeline_root_run_id = pipeline_root.info.run_id
        tags_dict = { 
                "preprocessor_name": ML_CONFIG.get("preprocessor_name"),
                "train_data_hash": output["hash"]["train_data_hash"],
                "test_data_hash": output["hash"]["test_data_hash"],
                "train_date_min": output["metadata"]["train_metadata"]["train_date_min"],
                "train_date_max": output["metadata"]["train_metadata"]["train_date_max"],
                "test_date_min": output["metadata"]["test_metadata"]["test_date_min"],
                "test_date_max": output["metadata"]["test_metadata"]["test_date_max"]
                }
        
        for key, value in FEATURE_CONFIG.items():
            tags_dict[key] = safe_tag_value(value)

        for key, value in DATA_CONFIG.items():
            tags_dict[key] = safe_tag_value(value)

        mlflow.set_tags(tags_dict)

        log_parquet(df = output["data"]["train_df"], filename=TRAIN_PATH, artifact_path="data")
        log_parquet(df=output["data"]["test_df"], filename=TEST_PATH, artifact_path="data")


        mlflow.log_artifact(args.data_config, artifact_path="config")
        mlflow.log_artifact(args.database_config, artifact_path="config")
        mlflow.log_artifact(args.feature_config, artifact_path="config")
        mlflow.log_artifact(args.ml_config, artifact_path="config")
        mlflow.log_artifact(args.search_space, artifact_path="config")
        
        search_space = get_search_space(
                args.search_space,
                model_name=ML_CONFIG.get("model_name"),
                optimizer_type=ML_CONFIG.get("optimizer_type")
            )

        final_model, best_cv_score, best_params, training_run_id = run_training_pipeline(
                                                                    X_train=X_train_preprocessed,
                                                                    y_train=y_train,
                                                                    ml_config=ML_CONFIG,
                                                                    trainer_cls=TimeSeriesTrainer,
                                                                    search_space = search_space,
                                                                    pipeline_root_run_id=pipeline_root_run_id,
                                                                    cat_feature_indices=cat_feature_indices)

        if final_model.has_feature_importance():
            importance_df = final_model.get_feature_importance(feature_names=feature_names)
            #importance_df = importance_df[importance_df["feature"] != "case_count_next_week"]
            log_parquet(df=importance_df, filename=FEATURE_IMPORTANCE_PATH,
                         artifact_path="feature_importance")

        metric_results = run_evaluation_pipeline(X_test=X_test_preprocessed,
                                y_test = y_test,
                                X_test_meta=output["test_meta"],
                                model = final_model,
                                best_cv_score=best_cv_score,
                                predictions_path=PREDICTIONS_PATH,
                                ml_config=ML_CONFIG,
                                pipeline_root_run_id=pipeline_root_run_id)
        

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


        shap_summary_path, shap_df= log_shap_summary(model_wrapper=final_model,
                                                     X_sample=X_test_preprocessed,
                                                     feature_names=feature_names,
                                                     shap_summary_path=SHAP_SUMMARY_PATH)

        log_parquet(df=shap_df, filename=SHAP_VALUES_PATH, artifact_path="explainability")
        mlflow.log_artifact(shap_summary_path, artifact_path="explainability")


if __name__ == "__main__":
    main()
    # import mlflow

    # experiment_name = "test_experiment"
    # experiment = start_mlflow_experiment(mlflow_uri=MLFLOW_URI,
    #                                      experiment_name=experiment_name,
                              
    #                                      )

    # mlflow.set_tracking_uri(MLFLOW_URI)
    
    # with mlflow.start_run():
    #     with open("test_s3.txt", "w") as f:
    #         f.write("testing S3 from pipeline")

    #     mlflow.log_artifact("test_s3.txt")
    
    # artifact_uri = mlflow.get_artifact_uri()
    # print("Artifact uri: ", artifact_uri)

    # import boto3
    # bucket = 'com.cg.cx-general'
    # prefix = 'mlflow/artifacts'
    # s3 = boto3.client("s3")
    # response = s3.list_objects_v2(
    #     Bucket=bucket,
    #     Prefix=prefix
    # )
    # logger.info(f"Checking S3 bucket: {bucket} | Prefix: {prefix}")
    # logger.info(f"Response: {response}")
    # for obj in response.get('Contents', []):
    #     logger.info(obj['Key']) 