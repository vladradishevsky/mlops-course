# lesson-05-modulation-isolation

Проект для обучения простой модели регрессии на погодных данных.

## Запуск

Из-за текущей структуры проекта модули из директории `src` нужно запускать с `PYTHONPATH`:

```bash
PYTHONPATH=. uv run python scripts/train.py
```

Скрипт обучит модель, посчитает `R^2` и сохранит модель в `models/model.pkl`.

Для запуска предсказаний:

```bash
PYTHONPATH=. uv run python scripts/predict.py
```

Для запуска через основную точку входа:

```bash
PYTHONPATH=. uv run python process.py
```

## Что делает пайплайн

- Загружает данные из `data/raw/data.csv`
- Удаляет строки с пропущенными значениями
- Добавляет признак `wind_humidity_ratio`
- Обучает `LinearRegression`
- Оценивает модель по метрике `R^2`
- Сохраняет модель в `models/model.pkl`
