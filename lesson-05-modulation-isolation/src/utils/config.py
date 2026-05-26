import json
from typing import Tuple
from pathlib import Path


def load_config(config_path: Path = Path("config.json")) -> dict:
    """
    Загружает конфигурационные параметры из JSON-файла.

    Args:
        config_path: Путь к конфигурационному файлу.

    Returns:
        Словарь с параметрами конфигурации.
    """
    with open(config_path) as f:
        config = json.load(f)

    return config


def get_data_paths(config: dict) -> Tuple[Path, Path]:
    """
    Извлекает пути к данным и модели из конфигурации.

    Args:
        config: Словарь с параметрами конфигурации.

    Returns:
        Кортеж из двух путей: путь к данным и путь для сохранения модели.
    """
    return Path(config["data_path"]), Path(config["output_model_path"])
