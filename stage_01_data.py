import mlflow
import pandas as pd
import pickle
from datetime import datetime

from pipelines.features_builder import build_features
from pipelines.data_builder import build_data
from db.engine import get_engine
from utils.mlflow_helpers import start_mlflow_experiment, log_git_to_mlflow, log_dvc_info
from utils.helpers import load_yaml_config
from config.env import DB_HOST, DB_USER, DB_NAME, DB_PASSWORD, DB_PORT, MLFLOW_URI
from config.filepaths import VILLAGE_EMBEDDINGS_PATH

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_config")
parser.add_argument("--database_config")
parser.add_argument("--feature_config")

args = parser.parse_args()

DATA_CONFIG = load_yaml_config(args.data_config)
DATABASE_CONFIG = load_yaml_config(args.database_config)
FEATURE_CONFIG = load_yaml_config(args.feature_config)

def main():
    os.makedirs("artifacts", exist_ok=True)

    engine = get_engine(
        db_user=DB_USER,
        db_password=DB_PASSWORD,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_name=DB_NAME
    )

    #  Features
    df = build_features(
        engine=engine,
        database_config=DATABASE_CONFIG,
        feature_config=FEATURE_CONFIG,
        village_embeddings_path=VILLAGE_EMBEDDINGS_PATH
    )

    df.to_parquet("artifacts/features.parquet")

    #  Data
    output = build_data(df=df, data_config=DATA_CONFIG)

    with open("artifacts/data.pkl", "wb") as f:
        pickle.dump(output, f)

    #  MLflow root run
    state = DATABASE_CONFIG.get('state').replace(" ", "_")
    disease = DATABASE_CONFIG.get('disease').replace(" ", "_")

    experiment_name = f"{state}_{disease}"

    mlflow.set_tracking_uri(MLFLOW_URI)
    start_mlflow_experiment(MLFLOW_URI, experiment_name)

    today = datetime.now().strftime("%Y/%m/%d")

    with mlflow.start_run(run_name=f"{experiment_name}_pipeline_root_{today}") as run:
        run_id = run.info.run_id

        log_git_to_mlflow()
        log_dvc_info()

        # save run_id for next stage
        with open("artifacts/run_id.txt", "w") as f:
            f.write(run_id)

        # log configs
        mlflow.log_artifact(args.data_config, artifact_path="config")
        mlflow.log_artifact(args.database_config, artifact_path="config")
        mlflow.log_artifact(args.feature_config, artifact_path="config")

if __name__ == "__main__":
    main()