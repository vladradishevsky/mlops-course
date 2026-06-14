import json
from typing import Dict


def load_config(path: str = "config.json") -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
