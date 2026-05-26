import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def evaluate_model(
    model: LinearRegression, X: pd.DataFrame, y: pd.Series
) -> float:
    """
    Вычисляет RMSE для оценки качества модели.

    Args:
        model: Обученная модель.
        X: Матрица признаков.
        y: Целевая переменная.

    Returns:
        Значение корня из среднеквадратичной ошибки (RMSE).
    """
    score = model.score(X, y)
    print(f"Model R^2 score: {score}")
    return score
