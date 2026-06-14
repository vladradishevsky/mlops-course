import logging
import sys
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_data
from src.features.engineering import add_wind_humidity_ratio
from src.models.evaluation import evaluate_model
from src.models.training import train_model
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main() -> None:
    config = load_config()

    setup_logger(config)
    logger = logging.getLogger(__name__)

    logger.info("Начало обучения модели")

    df = load_data(config["data"]["raw_path"])
    logger.info("Данные загружены")

    df = preprocess_data(df)
    logger.info("Данные предобработаны")

    df = add_wind_humidity_ratio(df)
    logger.info("Feature engineering выполнен")

    features = config["features"]["selected"] + config["features"]["engineered"]
    target = config["target"]

    X = df[features]
    y = df[target]

    model, X_test, y_test = train_model(
        X,
        y,
        random_state=config["model"]["random_state"],
        test_size=config["model"]["test_size"],
    )
    logger.info("Модель обучена")

    score = evaluate_model(model, X_test, y_test)
    logger.info(f"Model R^2 score: {score:.4f}")

    Path(config["model"]["output_path"]).parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, config["model"]["output_path"])
    logger.info(f"Модель сохранена в {config['model']['output_path']}")


if __name__ == "__main__":
    main()
