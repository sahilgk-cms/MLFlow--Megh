import pandas as pd
from typing import List
from features.schema import DATE_COL_WEEK_START, GROUP_COL

def create_rolling_features(df: pd.DataFrame, features: List[str], 
                              windows: List[int]) -> pd.DataFrame:
    """
    Creates rolling mean and std features for specified features per group.

    Args:
        df (pd.DataFrame): DataFrame with time-ordered data.
        group_col (str): Group identifier (e.g., 'taluk_name').
        date_col (str): Time column (e.g., 'week_start').
        features (list): Columns to apply rolling stats on.
        windows (list): List of window sizes (in weeks).

    Returns:
        pd.DataFrame: With additional rolling mean/std columns.
    """
    df = df.sort_values([GROUP_COL, DATE_COL_WEEK_START])
    for feat in features:
        for win in windows:
            df[f"{feat}_roll_mean_{win}w"] = (
                df.groupby(GROUP_COL)[feat].transform(lambda x: x.rolling(window=win, min_periods=win).mean())
            )
            df[f"{feat}_roll_std_{win}w"] = (
                df.groupby(GROUP_COL)[feat].transform(lambda x: x.rolling(window=win, min_periods=win).std())
            )
    return df