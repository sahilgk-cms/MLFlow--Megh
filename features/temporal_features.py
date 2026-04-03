import numpy as np
import pandas as pd
from features.schema import DATE_COL_WEEK_START

def add_month_sin_cos(df: pd.DataFrame, 
                      inplace: bool = False) -> pd.DataFrame:
    """
    Add cyclical month features to df: 'month_sin', 'month_cos'.
    Does NOT drop the original date_col.
    If inplace=False returns a new DataFrame; else modifies df and returns it.
    """
    if not inplace:
        df = df.copy()
    # ensure datetime (non-destructive if already datetime)
    df[DATE_COL_WEEK_START] = pd.to_datetime(df[DATE_COL_WEEK_START])
    month = df[DATE_COL_WEEK_START].dt.month.astype(int)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    return df