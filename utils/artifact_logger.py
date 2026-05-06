import mlflow
import pandas as pd
import os

def log_parquet(df: pd.DataFrame, filename: str, artifact_path: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True) 
    df.to_parquet(filename, index=False)
    mlflow.log_artifact(filename, artifact_path)

def log_config(config: dict, artifact_file: str):
    mlflow.log_dict(config, artifact_file)