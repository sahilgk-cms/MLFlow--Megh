import mlflow
import pandas as pd

def log_parquet(df: pd.DataFrame, filename: str, artifact_path: str):
    df.to_parquet(filename, index=False)
    mlflow.log_artifact(filename, artifact_path)

def log_config(config: dict, artifact_file: str):
    mlflow.log_dict(config, artifact_file)