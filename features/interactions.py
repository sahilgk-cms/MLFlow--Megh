import pandas as pd
from features.schema import ECO_PROB_COLS

def add_weather_interactions(df: pd.DataFrame,
                             interaction_lag: int,
                             precip_threshold: int,
                             humidity_threshold: int,
                             temp_threshold: int,
                             diurnal_threshold: int) -> pd.DataFrame:

    # Create binary variable for precipitation
    df['High_Precip'] = (df[f'total_precipitation_sum_mm_lag_{interaction_lag}'] >= precip_threshold).astype(int)

    # Create binary variable for relative humidity
    df['High_Humidity'] = (df[f'relative_humidity_percent_lag_{interaction_lag}'] > humidity_threshold).astype(int)

    # Define interaction variable
    df['High_Precip_Humidity'] = df['High_Precip'] * df['High_Humidity']


    df['temperature_mean'] = (df[f'temperature_2m_mean_celsius_lag_{interaction_lag}'] >= temp_threshold).astype(int)
    df['diurnal_high'] = (df[f'diurnal_lag_{interaction_lag}'] >= diurnal_threshold).astype(int)

    # Define interaction variable
    df['High_temp_Humidity'] = df['temperature_mean'] * df['High_Humidity']
    df['High_temp_Humidity_preci'] = df['temperature_mean'] * df['High_Humidity'] * df['High_Precip']

    return df


def eco_col_interactions(df: pd.DataFrame, 
                          interaction_lag: int) -> pd.DataFrame:
    for col in ECO_PROB_COLS:
        df[f'{col}_High_Precip'] = df[col] * df[f'total_precipitation_sum_mm_lag_{interaction_lag}']
        df[f'{col}_High_Humidity'] = df[col] * df[f'relative_humidity_percent_lag_{interaction_lag}']
        df[f'{col}_temperature_mean'] = df[col] * df[f'temperature_2m_mean_celsius_lag_{interaction_lag}']
    return df

def pca_col_interactions(df: pd.DataFrame,
                         interaction_lag: int) -> pd.DataFrame:
    pca_cols = [col for col in df.columns if col.startswith('PC')]

    for pca in pca_cols:
        df[f"{pca}_High_Precip"] =  df[pca] * df[f'total_precipitation_sum_mm_lag_{interaction_lag}']
        df[f"{pca}_High_Humidity"] =  df[pca] * df[f'relative_humidity_percent_lag_{interaction_lag}']
        df[f"{pca}_temperature_mean"] =  df[pca] * df[f'temperature_2m_mean_celsius_lag_{interaction_lag}']

    return df