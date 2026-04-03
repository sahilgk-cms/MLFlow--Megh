
import pandas as pd
from typing import List
from features.schema import DATE_COL_WEEK_START, GROUP_COL, CASE_COL

def shift_cases_forward(df, shift_by: int):
    """
    Shifts case counts forward by `shift_by` weeks for each group.
    Used to align weather(t) → cases(t+1).

    Args:
        df (pd.DataFrame): Aggregated data.
        shift_by (int): Number of weeks to shift forward.

    Returns:
        pd.DataFrame: DataFrame with shifted target column.
    """
    df = df.sort_values([GROUP_COL, DATE_COL_WEEK_START])
    df[f'{CASE_COL}_next_week'] = (
        df.groupby(GROUP_COL)[CASE_COL].shift(-shift_by)
    )
    return df


def create_lag_features(df: pd.DataFrame, features: List[str], 
                        lags: List[int],) -> pd.DataFrame:
    """
    Create lag features for the specified columns per group.

    Args:
        df (pd.DataFrame): Time-indexed data.
        features (list): Feature columns to lag.
        lags (list): List of lag values in weeks.

    Returns:
        pd.DataFrame: With lagged columns added.
    """
    df = df.sort_values([GROUP_COL, DATE_COL_WEEK_START])
    for feat in features:
        for lag in lags:
            df[f"{feat}_lag_{lag}"] = df.groupby(GROUP_COL)[feat].shift(lag)
    return df


def fill_lag_values(df: pd.DataFrame) -> pd.DataFrame:
    
    lagged_cols = [
        col for col in df.columns
        if 'lag_' in col or 'roll_' in col
    ]

    df = df.sort_values([GROUP_COL, DATE_COL_WEEK_START])

    for col in lagged_cols:
        df[col] = (
            df
            .groupby(GROUP_COL)[col]
            .transform(lambda x: x.bfill().ffill())
        )
    return df


