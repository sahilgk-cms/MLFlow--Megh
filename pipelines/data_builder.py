from data.train_test_split import temporal_train_test_split, sort_data, drop_null_values
from data.data_hash import get_data_hash
from data.metadata import extract_data_metadata
import pandas as pd
from data.split_features_target import split_features_target



def build_data(df: pd.DataFrame,  data_config: dict) -> dict:

    train_df, test_df = temporal_train_test_split(df=df, cutoff_week=data_config.get("cutoff_week"))
    train_df, test_df = sort_data(train_df=train_df, test_df=test_df)

    train_df, test_df = drop_null_values(train_df=train_df,
                                         test_df=test_df)

    train_data_hash = get_data_hash(train_df)
    test_data_hash = get_data_hash(test_df)

    train_metadata = extract_data_metadata(train_df, train=True)
    test_metadata = extract_data_metadata(test_df)
  
    X_train, y_train = split_features_target(train_df)
    X_test, y_test, X_test_meta = split_features_target(test_df, return_meta=True )
    
    return {
        "data": {
            "train_df": train_df,
            "test_df": test_df
        },
        "features": {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        },
        "metadata": {
            "train_metadata": train_metadata,
            "test_metadata": test_metadata
        },
        "hash": {
            "train_data_hash": train_data_hash,
            "test_data_hash": test_data_hash
        },
        "test_meta": X_test_meta
    }



