# src/models/training.py
from typing import Dict, Optional

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def train_model(
    X: pd.DataFrame, y: pd.Series, model_params: Optional[Dict] = None
) -> LinearRegression:
    """
    Обучает модель случайного леса на переданных данных.

    Args:
        X: Матрица признаков.
        y: Целевая переменная.
        model_params: Параметры модели, например, random_state, n_estimators.

    Returns:
        Обученная модель RandomForestRegressor.
    """
    if not model_params:
        model_params = {}

    model = LinearRegression(**model_params)
    model.fit(X, y)
    return model
