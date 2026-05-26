from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.data.ingestion import load_data
from src.data.preprocessing import clean_data
from src.features.base_features import add_base_features
from src.models.evaluation import evaluate_model
from src.models.training import train_model
from src.utils.config import get_data_paths, load_config


def train():
    """
    Обучает модель машинного обучения на основе конфигурации.
    Последовательно выполняет:
      1. Загрузка данных
      2. Очистка данных
      3. Добавление новых признаков
      4. Обучение модели
      5. Сохранение модели
      6. Оценка качества модели
    """
    config: dict = load_config()
    data_path: Path
    output_path: Path
    data_path, output_path = get_data_paths(config)

    # Загрузка и предобработка
    df: pd.DataFrame = load_data(data_path)
    df = clean_data(df)
    df = add_base_features(df)

    features: list = config["features"] + ["wind_humidity_ratio"]
    X: pd.DataFrame = df[features]
    y: pd.Series = df[config["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=config["random_state"]
    )

    # Обучение
    model: LinearRegression = train_model(X_train, y_train, config.get("model_params", {}))

    # Сохранение
    joblib.dump(model, output_path)

    # Оценка качества
    evaluate_model(model, X_test, y_test)

    print("Модель обучена и успешно сохранена.")


if __name__ == "__main__":
    train()
