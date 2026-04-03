import pandas as pd
from typing import List, Tuple, Dict
from features.schema import DATE_COL, GROUP_COL

def calculate_rolling_max_streak(binary_series, window):
    """
    Vectorized approach to find max consecutive days within a rolling window.
    """
    # Calculate streaks: [0, 1, 1, 0, 1] -> [0, 1, 2, 0, 1]
    streak = binary_series * (binary_series.groupby((binary_series != binary_series.shift()).cumsum()).cumcount() + 1)
    # The max streak within the window
    return streak.rolling(window=window, min_periods=1).max()

def calculate_weather_metrics(df, buckets: Dict[str, List[tuple]],
                               window: int) -> pd.DataFrame:
    
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.sort_values([GROUP_COL, DATE_COL]).copy()

    # Define buckets for your specific column names
    # Adjust these ranges if needed based on your local climate

    for feat in buckets.keys():
        if feat not in df.columns:
            continue

        feat_buckets = buckets.get(feat, [])
        for (low, high) in feat_buckets:
            col_base = f"{feat}_{low}_{high}"
            # 1. Create the indicator column in the dataframe
            # This was the missing link in your previous run
            indicator_col = f"is_{col_base}"
            df[indicator_col] = df[feat].between(low, high, inclusive = 'left').astype(int)

            # 2. Total Days (Rolling Sum)
            df[f"{col_base}_total_days"] = df.groupby('sub_district')[indicator_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )

            # 3. Max Consecutive Days (Vectorized Logic)
            # We calculate the "streak" for each row and then take the rolling max
            df[f"{col_base}_max_conseq"] = df.groupby('sub_district')[indicator_col].transform(
                lambda x: calculate_rolling_max_streak(x, window)
            )

            # Optional: Remove the temporary indicator column to keep it clean
            df.drop(columns=[indicator_col], inplace=True)

    return df
