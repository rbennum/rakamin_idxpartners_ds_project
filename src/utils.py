import pandas as pd
import numpy as np
import joblib
import os

from datetime import datetime
from typing import Optional

from sklearn.model_selection import GridSearchCV


def skim_data(data) -> pd.DataFrame:
    """
    Skims the dataframe for features, feature types, null, negative, and zero values percentage, and the number of unique values.

    :param DataFrame data: The input dataframe.
    :return: A dataframe contains the summary of the input dataframe.
    :rtype: DataFrame
    """

    numeric_cols = set(data.select_dtypes(include=[np.number]).columns)
    numeric_stats = {}

    for col in numeric_cols:
        numeric_stats[col] = {
            "neg_%": round((data[col] < 0).mean() * 100, 3),
            "zero_%": round((data[col] == 0).mean() * 100, 3),
        }

    skimmed_data = pd.DataFrame(
        {
            "feature": data.columns.values,
            "dtype": data.dtypes.astype(str).values,
            "null_%": round(data.isna().mean() * 100, 3).values,
            "negative_%": [
                numeric_stats.get(col, {}).get("neg_%", "-") for col in data.columns
            ],
            "zero_%": [
                numeric_stats.get(col, {}).get("zero_%", "-") for col in data.columns
            ],
            "n_unique": data.nunique().values,
            "unique_%": round(data.nunique() / len(data) * 100, 2).values,
            "sample_values": [
                list(data[col].dropna().unique()[:5]) for col in data.columns
            ],
        }
    )
    print(f"Total duplicate rows: {data.duplicated().sum()}")
    print(f"DF shape: {data.shape}")
    return skimmed_data


def save_model(model_object, filename, directory="models"):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Folder at '{directory}' has been created.")

        filepath = os.path.join(directory, filename)
        joblib.dump(model_object, filepath)
        print(f"Model object successfully saved at: {filepath}")
    except Exception as e:
        print(f"Error when trying to save the object: {e}")


def load_model(filepath) -> Optional[GridSearchCV]:
    try:
        loaded_object = joblib.load(filepath)
        print(f"Model loaded from: {filepath}")
        return loaded_object
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"Error when loading object: {e}")
        return None


def get_time(time: Optional[datetime] = None) -> str:
    if time is None:
        time = datetime.now()
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    return timestamp_str
