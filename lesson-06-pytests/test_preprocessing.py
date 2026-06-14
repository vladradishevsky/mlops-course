import pandas as pd
import pytest

from preprocessing import preprocess_data


@pytest.fixture
def sample_rides() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fare_amount": [10.0, -5.0, 30.0, 15.0],
            "passenger_count": [1, 2, 7, 3],
            "pickup_longitude": [-73.90, -73.95, -73.98, -73.99],
            "pickup_latitude": [40.70, 40.72, 40.74, 40.75],
            "dropoff_longitude": [-74.00, -73.90, -73.97, -73.95],
            "dropoff_latitude": [40.80, 40.70, 40.78, 40.78],
        }
    )


def test_preprocess_data_filters_invalid_rows_and_returns_features(
    sample_rides: pd.DataFrame,
) -> None:
    result = preprocess_data(sample_rides.copy())

    assert list(result.columns) == ["distance", "passenger_count"]
    assert list(result.index) == [0, 3]
    assert result["passenger_count"].tolist() == [1, 3]
    assert result["distance"].tolist() == pytest.approx([0.1414213562, 0.05])


def test_preprocess_data_does_not_modify_input_dataframe(
    sample_rides: pd.DataFrame,
) -> None:
    original = sample_rides.copy(deep=True)

    preprocess_data(sample_rides)

    pd.testing.assert_frame_equal(sample_rides, original)
