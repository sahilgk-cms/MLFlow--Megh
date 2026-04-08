import pandas as pd
from typing import Tuple
from data.schema import DATE_COL_WEEK_START, TARGET_COL

def temporal_train_test_split(df: pd.DataFrame,
                              cutoff_week: int,
                              date_col=DATE_COL_WEEK_START) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_date = df[date_col].max() - pd.Timedelta(weeks=cutoff_week)
    train_df = df[df[date_col] <= cutoff_date].copy()
    test_df = df[df[date_col] > cutoff_date].copy()
    return train_df, test_df


def sort_data(train_df: pd.DataFrame,
              test_df: pd.DataFrame,
              date_col=DATE_COL_WEEK_START) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.sort_values(date_col).reset_index(drop=True)
    test_df = test_df.sort_values(date_col).reset_index(drop=True)
    return train_df, test_df


def drop_null_values(train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                     target_col=TARGET_COL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_null_idx = train_df[train_df[target_col].isnull()].index
    train_df = train_df.drop(index = train_null_idx).reset_index(drop = True)

    test_null_idx = test_df[test_df[target_col].isnull()].index
    test_df = test_df.drop(index=test_null_idx).reset_index(drop=True)

    return train_df, test_df

