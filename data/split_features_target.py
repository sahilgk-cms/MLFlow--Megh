import pandas as pd
from typing import Tuple
from data.schema import DATE_COL_WEEK_START, TARGET_COL, TEST_META_COLS

def split_features_target(df: pd.DataFrame,
                          target_col = TARGET_COL,
                          date_col = DATE_COL_WEEK_START,
                          return_meta:bool = False,
                          test_meta_cols = TEST_META_COLS)-> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col, date_col], errors="ignore")
    y = df[target_col]

    if return_meta:
        X_meta = df[test_meta_cols]
        return X, y, X_meta

    return X, y