import pandas as pd
from data.schema import DATE_COL_WEEK_START

def extract_data_metadata(df: pd.DataFrame, date_col = DATE_COL_WEEK_START, train: bool = False) -> dict:
    if train:
        return {
            "train_records": len(df),
            "train_date_min": df[date_col].min().strftime("%Y-%m-%d"),
            "train_date_max": df[date_col].max().strftime("%Y-%m-%d")
        }
    else:
        return {
            "test_records": len(df),
            "test_date_min": df[date_col].min().strftime("%Y-%m-%d"),
            "test_date_max": df[date_col].max().strftime("%Y-%m-%d")
        }