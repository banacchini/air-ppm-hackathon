import pytest
import pandas as pd
import numpy as np

from prediction import forecast_with_lag


# Mock model with feature_names_ to simulate CatBoost
class DummyModel:
    def __init__(self):
        self.feature_names_ = [
            'year', 'month', 'day', 'hour',
            'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
            'weekday_sin', 'weekday_cos', 'station_code',
            'pm10_lag_1', 'pm10_lag_2'
        ]
    def predict(self, X):
        return np.array([42.0])  # always predict 42

dummy_model = DummyModel()

# --- Fixtures ---
@pytest.fixture
def minimal_case():
    """A minimal valid case dict for testing forecast_with_lag."""
    return {
        "case_id": "case_test",
        "stations": [
            {
                "station_code": "StationX",
                "longitude": 10.0,
                "latitude": 50.0,
                "history": [
                    {"timestamp": "2025-01-01T00:00:00", "pm10": 40.0},
                    {"timestamp": "2025-01-01T01:00:00", "pm10": 42.0},
                ],
            }
        ],
        "target": {
            "longitude": 10.0,
            "latitude": 50.0,
            "prediction_start_time": "2025-01-01T02:00:00",
        },
        "weather": [
            {"date": "2025-01-01T00:00:00", "tmp": "+0050,1", "wnd": "260,1,N,0030,1"}
        ],
    }
# --- Tests ---

def test_happy_path(minimal_case):
    """Valid case should return 24 forecast rows without errors."""
    result = forecast_with_lag(minimal_case, dummy_model, horizon=24)
    assert result["case_id"] == "case_test"
    assert len(result["forecast"]) == 24
    assert all("pm10_pred" in row for row in result["forecast"])

def test_missing_history(minimal_case):
    """Error if station history is missing."""
    minimal_case["stations"][0]["history"] = []
    with pytest.raises(IndexError):
        forecast_with_lag(minimal_case, dummy_model)

def test_not_enough_history(minimal_case):
    """Error if only one history point is available."""
    minimal_case["stations"][0]["history"] = [
        {"timestamp": "2024-12-31T23:00:00", "pm10": 20}
    ]
    with pytest.raises(IndexError):
        forecast_with_lag(minimal_case, dummy_model)

def test_missing_station_code(minimal_case):
    """Error if station_code key is missing."""
    del minimal_case["stations"][0]["station_code"]
    with pytest.raises(KeyError):
        forecast_with_lag(minimal_case, dummy_model)

def test_invalid_prediction_time(minimal_case):
    """Error if prediction_start_time is invalid."""
    minimal_case["target"]["prediction_start_time"] = "not-a-date"
    with pytest.raises(ValueError):
        forecast_with_lag(minimal_case, dummy_model)

