WEATHER_COLS = ['date', 'sub_district', 'district', 'state', 'dewpoint_temperature',
        'maximum_temperature', 'mean_temperature', 'minimum_temperature',
        'relative_humidity', 'total_precipitation', 'confirmed_diagnosis',
        'no_of_cases']

MIN_DATE = '2022-12-31'

RENAME_MAPPING = {
       'minimum_temperature': 'temperature_2m_min_celsius',
       'maximum_temperature': 'temperature_2m_max_celsius',
       'mean_temperature': 'temperature_2m_mean_celsius',
       'dewpoint_temperature': 'temperature_2m_dewpoint_celsius',
       'relative_humidity': 'relative_humidity_percent',
       'total_precipitation': 'total_precipitation_sum_mm'

}

LULC_COLS = ['water', 'trees',
       'flooded_vegetation', 'crops', 'built_area', 'bare_ground', 'snow_ice',
       'clouds', 'rangeland']

DATE_COL = "date"
GROUP_COL = "sub_district"
DATE_COL_WEEK_START = "week_start"





CASE_COL = 'no_of_cases'
CASE_COL_LAG_2 = "no_of_cases_lag_2"


STATIC_COLS = ['dist_name', "agro_zones",  "rural_pop_density_per_sqkm", 'water', 'trees',
       'flooded_vegetation', 'crops', 'built_area', 'bare_ground', 'snow_ice',
       'clouds', 'rangeland',"urban_pop_density_per_sqkm", "rural_population", "urban_population", "total_population"]


STATIC_COLS_2 = ['water', 'trees',
       'flooded_vegetation', 'crops', 'built_area', 'bare_ground', 'snow_ice',
       'clouds', 'rangeland', "rural_population", "urban_population", "total_population"]

STATIC_COLS_3 = ['rural_pop_density_per_sqkm', 'urban_pop_density_per_sqkm']

TEMPORAL_COLS = ['temperature_2m_mean_celsius', 'relative_humidity_percent',
        'diurnal']

RAIN_COLS = ['total_precipitation_sum_mm']

ECO_PROB_COLS = [
    'eco_prob_0', 'eco_prob_1', 'eco_prob_2',
    'eco_prob_3', 'eco_prob_4', 'eco_prob_5'
]
