# src/preprocessing.py
import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Выполняет предобработку данных о поездках Uber."""
    # 1. Фильтрация по стоимости и пассажирам
    mask = (
        (df["fare_amount"] > 0)
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
    )

    df_cleaned = df.drop(index=df.loc[~mask].index, inplace=False)

    # 2. Создание нового признака 'distance'
    df_cleaned["distance"] = np.sqrt(
        (df_cleaned["dropoff_longitude"] - df_cleaned["pickup_longitude"]) ** 2
        + (df_cleaned["dropoff_latitude"] - df_cleaned["pickup_latitude"]) ** 2
    )
    final_features = df_cleaned[["distance", "passenger_count"]]
    return final_features

