import pandas as pd


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет базовые признаки в DataFrame.

    Args:
        df: Исходный DataFrame.

    Returns:
        DataFrame с новыми признаками.
    """
    df = df.copy()
    df["wind_humidity_ratio"] = df["wind_speed"] / df["humidity"].replace(0, 0.1)
    return df
