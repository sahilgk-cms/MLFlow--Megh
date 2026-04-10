import os

CURRENT_DIR = os.getcwd()

LOGS_DIRECTORY = os.path.join(CURRENT_DIR, "logs")

SEARCH_SPACE_PATH = os.path.join(CURRENT_DIR, "config", "search_spaces.yml")
VILLAGE_EMBEDDINGS_PATH = os.path.join(CURRENT_DIR, "embeddings", "emb_village.csv")





ARTIFACTS_FOLDER = os.path.join(CURRENT_DIR, "artifacts")
FEATURES_ARTIFACT = os.path.join(ARTIFACTS_FOLDER, "features.parquet")
DATA_ARTIFACT = os.path.join(ARTIFACTS_FOLDER, "data.pkl")
RUN_ARTIFACT = os.path.join(ARTIFACTS_FOLDER, "run_id.txt")
TRAIN_PATH = os.path.join(ARTIFACTS_FOLDER, "train_dataset.parquet")
TEST_PATH = os.path.join(ARTIFACTS_FOLDER, "test_dataset.parquet")
FEATURE_IMPORTANCE_PATH = os.path.join(ARTIFACTS_FOLDER, "feature_importance.parquet")
PREDICTIONS_PATH = os.path.join(ARTIFACTS_FOLDER, "predictions.parquet")
SHAP_VALUES_PATH = os.path.join(ARTIFACTS_FOLDER, "shap_values.parquet")
SHAP_SUMMARY_PATH =  os.path.join(ARTIFACTS_FOLDER, "shap_summary.png")
