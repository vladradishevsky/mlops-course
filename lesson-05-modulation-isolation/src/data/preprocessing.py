import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет строки с пропущенными значениями.

    Args:
        df: Исходный DataFrame.

    Returns:
        Очищенный DataFrame без пропусков.
    """
    return df.dropna()
