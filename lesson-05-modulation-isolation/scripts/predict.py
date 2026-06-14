import logging
import sys
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.ingestion import load_data
from src.features.engineering import add_wind_humidity_ratio
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main() -> None:
    config = load_config()

    setup_logger(config)
    logger = logging.getLogger(__name__)
    logger.info("Начало предсказания")

    df = load_data(config["data"]["raw_path"])
    logger.info("Данные загружены")

    df = add_wind_humidity_ratio(df)
    logger.info("Feature engineering выполнен")

    features = config["features"]["selected"] + config["features"]["engineered"]
    X = df[features]

    model = joblib.load(config["model"]["output_path"])
    predictions = model.predict(X)

    logger.info(f"Сделано предсказаний: {len(predictions)}")
    print(predictions)


if __name__ == "__main__":
    main()
