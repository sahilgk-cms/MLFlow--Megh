import pandas as pd
from features.schema import RENAME_MAPPING, DATE_COL, DATE_COL_WEEK_START, GROUP_COL, MIN_DATE, ECO_PROB_COLS


def clean_and_merge_statewise_cases_and_weather_data(statewise_data: pd.DataFrame,
                                                     weather_data: pd.DataFrame,
                                                     min_date = MIN_DATE) -> pd.DataFrame:
    statewise_data[DATE_COL] = pd.to_datetime(statewise_data[DATE_COL], format= "%Y/%m/%d")
    statewise_data = statewise_data[statewise_data[DATE_COL] > min_date]

    weather_data[DATE_COL] = pd.to_datetime(weather_data[DATE_COL], format= "%Y/%m/%d")
    statewise_data = weather_data.merge(statewise_data, 
                                        on= ['date', 'sub_district', 'district', 'state'], 
                                        how = 'inner')
    
    # high_case_subdists = (
    #                         statewise_data.groupby(['district', 'sub_district'])['no_of_cases']
    #                         .sum()
    #                         .reset_index()
    #                         .query('no_of_cases > 10')
    #                     )
    # statewise_data = statewise_data[statewise_data['sub_district'].isin(high_case_subdists)]
    statewise_data = statewise_data.rename(columns=RENAME_MAPPING)
    statewise_data['diurnal'] = statewise_data['temperature_2m_max_celsius'] - statewise_data['temperature_2m_min_celsius']
    statewise_data[DATE_COL] = pd.to_datetime(statewise_data[DATE_COL], format="%Y-%m-%d")
    statewise_data[DATE_COL_WEEK_START] = statewise_data[DATE_COL] - pd.to_timedelta(statewise_data[DATE_COL].dt.weekday, unit='D')
    statewise_data['year'] = statewise_data[DATE_COL_WEEK_START].dt.year
    return statewise_data


def merge_statewise_cases_and_temporal(statewise_temporal: pd.DataFrame,
                                       statewise_cases: pd.DataFrame):
    statewise_temporal = statewise_temporal.merge(statewise_cases, 
                                                  on= [GROUP_COL, DATE_COL_WEEK_START], how= 'left')
    statewise_temporal['year'] = statewise_temporal[DATE_COL_WEEK_START].dt.year
    return statewise_temporal


def merge_statewise_and_lulc(statewise_temporal: pd.DataFrame,
                       df_lulc: pd.DataFrame) -> pd.DataFrame:
    df_lulc['sub_district'] = (
                            df_lulc['sub_district']
                            .astype(str)          # ensure string type
                            .str.strip()          # remove leading/trailing spaces
                            .str.title()          # First letter capital, rest small
                        )
    statewise_final= statewise_temporal.merge(df_lulc, on= ['year', 'sub_district'], how = 'left')
    return statewise_final


def load_and_preprocess_village_embeddings(filepath: str) -> pd.DataFrame:
    village_emb = pd.read_csv(filepath)
    pca_cols = [col for col in village_emb.columns if col.startswith('PC') or col.startswith('eco_prob')]
    village_emb_sub = village_emb.groupby(['subdistric'])[pca_cols].mean().reset_index()

    village_emb_sub['ecocluster'] = (
        village_emb_sub[ECO_PROB_COLS]
        .idxmax(axis=1)
        .str.extract('(\d+)')
        .astype(int)
    )
    return village_emb_sub


def merge_statewise_and_village_emb(statewise_final: pd.DataFrame,
                                    village_emb_sub: pd.DataFrame) -> pd.DataFrame:
    statewise_final = statewise_final.merge(village_emb_sub, 
                                            left_on="sub_district",
                                            right_on="subdistric",
                                            how="left")
    return statewise_final


def merge_statewise_final_and_statewise_new(statewise_final: pd.DataFrame,
                                             statewise_new: pd.DataFrame) -> pd.DataFrame:
    statewise_final = statewise_final.drop(columns='subdistric')
    statewise_final = statewise_final.merge(statewise_new,
                                            on=[GROUP_COL, DATE_COL_WEEK_START],
                                            how='inner')
    return statewise_final
    
