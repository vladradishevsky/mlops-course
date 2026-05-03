from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from config import ModelParams, TrainingConfig


def load_config(path: Path = Path("config.json")) -> TrainingConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    return TrainingConfig.model_validate_json(path.read_text(encoding="utf-8"))


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    return pd.read_csv(path)


def preprocess(
    data: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    required_columns = [*config.features, config.target_column]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    clean_data = data.dropna()
    if clean_data.empty:
        raise ValueError("Dataset is empty after dropping missing values")

    X = clean_data[config.features]
    y = clean_data[config.target_column]

    return X, y


def train(
    X: pd.DataFrame,
    y: pd.Series,
    params: ModelParams,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        random_state=params.random_state,
    )
    model.fit(X, y)
    return model


def save_model(model: RandomForestRegressor, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(model, f)


def main() -> None:
    config = load_config()
    data = load_data(config.data_path)
    X, y = preprocess(data, config)

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        random_state=config.model_params.random_state,
    )

    model = train(X_train, y_train, config.model_params)
    save_model(model, config.output_path)

    print("Model saved successfully")


if __name__ == "__main__":
    main()
