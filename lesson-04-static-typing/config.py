from pathlib import Path

from pydantic import BaseModel, Field


class ModelParams(BaseModel):
    n_estimators: int = Field(gt=0)
    max_depth: int = Field(gt=0)
    random_state: int


class TrainingConfig(BaseModel):
    data_path: Path
    target_column: str = Field(min_length=1)
    features: list[str] = Field(min_length=1)
    model_params: ModelParams
    output_path: Path
