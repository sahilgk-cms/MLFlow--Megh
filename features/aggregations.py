import pandas as pd
from features.schema import DATE_COL_WEEK_START, GROUP_COL, TEMPORAL_COLS, RAIN_COLS, CASE_COL


def aggregate_to_weekly(df: pd.DataFrame):
    total_days_cols = [col for col in df.columns if 'total_days' in col]
    max_conseq_cols = [col for col in df.columns if 'max_conseq' in col]
    total_cols = total_days_cols + max_conseq_cols
    
    # Aggregate: take LAST value in each week
    weekly = (
        df.groupby([GROUP_COL, DATE_COL_WEEK_START], as_index=False)
          .agg(
              {
                  **{col: 'last' for col in total_cols},
                  
              }
          )
    )

    return weekly


def aggregate_weekly_mean(df: pd.DataFrame):
    """
    Aggregates temporal features to weekly resolution per group.

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Aggregated weekly values per group.
    """

    weekly = (
        df.groupby([GROUP_COL, DATE_COL_WEEK_START])[TEMPORAL_COLS]
        .mean()
        .reset_index()
        .sort_values([GROUP_COL, DATE_COL_WEEK_START])
    )
    return weekly


def aggregate_weekly_sum(df):
    """
    Aggregates temporal features to weekly resolution per group.

    Args:
        df (pd.DataFrame): Input data.
      

    Returns:
        pd.DataFrame: Aggregated weekly values per group.
    """

    features = RAIN_COLS + [CASE_COL]
    weekly = (
        df.groupby([GROUP_COL, DATE_COL_WEEK_START])[features]
        .sum()
        .reset_index()
        .sort_values([GROUP_COL, DATE_COL_WEEK_START])
    )
    return weekly

