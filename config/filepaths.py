import os

CURRENT_DIR = os.getcwd()
LOGS_DIRECTORY = os.path.join(CURRENT_DIR, "logs")

SEARCH_SPACE_PATH = os.path.join(CURRENT_DIR, "config", "search_spaces.yml")
VILLAGE_EMBEDDINGS_PATH = os.path.join(CURRENT_DIR, "embeddings", "emb_village.csv")


TRAIN_PATH = "train_dataset.parquet"
TEST_PATH = "test_dataset.parquet"
FEATURE_IMPORTANCE_PATH = "feature_importance.parquet"
PREDICTIONS_PATH = "predictions.parquet"
SHAP_VALUES_PATH = "shap_values.parquet"
SHAP_SUMMARY_PATH =  "shap_summary.png"

DATABASE_CONFIG_PATH = "config/database_config.json"
FEATURE_CONFIG_PATH = "config/feature_config.json"
DATA_CONFIG_PATH = "config/data_config.json"
