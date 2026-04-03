import pandas as pd
from datetime import datetime
from data.schema import DATE_COL_WEEK_START, TARGET_COL, DISTRICT_COL, SUB_DISTRICT_COL

def build_prediction_data(predictions: pd.Series,
                          X_test_meta: pd.DataFrame,
                          best_cv_score: float,
                          test_score: float,
                          metric_name: str) -> pd.DataFrame:
    prediction_df = pd.DataFrame({
        DATE_COL_WEEK_START: X_test_meta[DATE_COL_WEEK_START],
        'predicted_case_count': predictions,
        'prediction_date': datetime.now().strftime('%Y-%m-%d'),
        f'best_cv_{metric_name}': best_cv_score,
        f"test_{metric_name}": test_score,
        SUB_DISTRICT_COL: X_test_meta[SUB_DISTRICT_COL],
        #DISTRICT_COL: X_test_meta[DISTRICT_COL],
        TARGET_COL: X_test_meta[TARGET_COL]
    })
    return prediction_df
