from db.db_loader import load_cases_statewise, load_weather_data_statewise, load_lulc
from features.data_processing import clean_and_merge_statewise_cases_and_weather_data, merge_statewise_cases_and_temporal, merge_statewise_and_lulc, load_and_preprocess_village_embeddings, merge_statewise_and_village_emb, merge_statewise_final_and_statewise_new
from features.aggregations import aggregate_weekly_mean, aggregate_to_weekly, aggregate_weekly_sum
from features.lag_features import shift_cases_forward, create_lag_features, fill_lag_values
from features.rolling_features import create_rolling_features
from features.interactions import add_weather_interactions, eco_col_interactions, pca_col_interactions
from features.temporal_features import add_month_sin_cos
from features.weather_processing import calculate_weather_metrics
#from db.db_loader import append_df_to_db
from features.schema import TEMPORAL_COLS, RAIN_COLS, CASE_COL, CASE_COL_LAG_2
import sqlalchemy
import pandas as pd
import argparse


def build_features(engine: sqlalchemy.engine.base.Engine, database_config: dict,  
                   feature_config: dict, village_embeddings_path: str) -> pd.DataFrame:

    print("Loading statewise and weather data...")
    statewise_data = load_cases_statewise(engine, state=database_config.get("state"),
                                          disease=database_config.get("disease"))
    print(f"statewise_Data: {statewise_data.shape}")
    weather_data = load_weather_data_statewise(engine, state=database_config.get("state"))
    print(f"weather_Data: {weather_data.shape}")
    statewise_data = clean_and_merge_statewise_cases_and_weather_data(statewise_data, weather_data)


    # print("Calculating weather metrics...")
    statewise_data = calculate_weather_metrics(statewise_data, buckets=feature_config.get("bucket_defs"),
                                               window=feature_config.get("window"))


    # print("Aggragating...")
    statewise_new = aggregate_to_weekly(statewise_data)

    statewise_temporal = aggregate_weekly_mean(statewise_data)

    statewise_cases = aggregate_weekly_sum(statewise_data)

    statewise_temporal = merge_statewise_cases_and_temporal(statewise_temporal, statewise_cases)


    # print("Lag features...")
    statewise_temporal = shift_cases_forward(statewise_temporal, shift_by=feature_config.get("shift_by"))

    statewise_temporal = create_lag_features(statewise_temporal, features=TEMPORAL_COLS+RAIN_COLS,
                                              lags=feature_config.get("lags_weather"))
    statewise_temporal = create_lag_features(statewise_temporal, features=[CASE_COL], 
                                             lags=feature_config.get("lags_cases"))
    
    # print("Rolling features...")
    statewise_temporal = create_rolling_features(statewise_temporal, features=TEMPORAL_COLS+RAIN_COLS,
                                                 windows=feature_config.get("rolling_windows") )
    statewise_temporal = create_rolling_features(statewise_temporal, features=[CASE_COL_LAG_2],
                                                 windows=feature_config.get("rolling_windows") )

    # print("Locading lulcl and merging with statewise_temporal...")
    df_lulc = load_lulc(engine, state = database_config.get("state"))
    statewise_final = merge_statewise_and_lulc(statewise_temporal, df_lulc)

    # print("Weather interactions")
    statewise_final = add_weather_interactions(statewise_temporal,
                                               interaction_lag=feature_config.get("interaction_lag"),
                                               precip_threshold=feature_config.get("precip_threshold"),
                                               humidity_threshold=feature_config.get("humidity_threshold"),
                                               temp_threshold=feature_config.get("temp_threshold"),
                                               diurnal_threshold=feature_config.get("diurnal_threshold"))
    
    # print("Loading village embeddings and merging with statewise_final...")
    village_emb_sub = load_and_preprocess_village_embeddings(filepath=village_embeddings_path)
    statewise_final = merge_statewise_and_village_emb(statewise_final, village_emb_sub)

    # print("Eco and pca cols interactions...")
    statewise_final = eco_col_interactions(statewise_final,
                                           interaction_lag=feature_config.get("interaction_lag"))
    statewise_final = pca_col_interactions(statewise_final,
                                           interaction_lag=feature_config.get("interaction_lag"))
    
    # print("sin and cos...")
    statewise_final = add_month_sin_cos(statewise_final)

    # print("fill lag values and merge with new...")
    statewise_final = fill_lag_values(statewise_final)
    statewise_final = merge_statewise_final_and_statewise_new(statewise_final, statewise_new)
    #return statewise_data
    return statewise_final
