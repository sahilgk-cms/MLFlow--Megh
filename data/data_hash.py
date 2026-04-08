import pandas as pd
import hashlib

def get_data_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()
