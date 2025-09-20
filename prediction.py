import joblib
import pandas as pd
import numpy as np

cat_model = joblib.load("models/catboost_best_model.pkl")

def forecast_with_lag(case, model, horizon=24):
    """
    case: dict containing station info, history, target, weather
    model: trained CatBoost model
    horizon: number of hours to forecast (default: 24)
    """

    # Get station info
    station = case["stations"][0]   # assuming one station for now
    history = pd.DataFrame(station["history"])
    history["timestamp"] = pd.to_datetime(history["timestamp"])

    # Start from the last known PM10
    pm10_lag_1 = history["pm10"].iloc[-1]
    pm10_lag_2 = history["pm10"].iloc[-2]

    # Build the forecast rows
    start_time = pd.to_datetime(case["target"]["prediction_start_time"])
    forecast_rows = []

    for i in range(horizon):
        ts = start_time + pd.Timedelta(hours=i)
        year = ts.year
        month = ts.month
        day = ts.day
        weekday = ts.weekday  # Monday=0
        hour = ts.hour
        day_of_year = ts.dayofyear

        # Cyclic encoding for hour (24h cycle)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Cyclic encoding for day of year (seasonality)
        doy_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        doy_cos = np.cos(2 * np.pi * day_of_year / 365.25)

        # Optional: encode weekday cyclically
        weekday_sin = np.sin(2 * np.pi * weekday / 7)
        weekday_cos = np.cos(2 * np.pi * weekday / 7)

        columns = ['year', 'month', 'day', 'hour', 'hour_sin', 'hour_cos', 'doy_sin',
         'doy_cos', 'weekday_sin', 'weekday_cos', 'station_code']

        # --- Build input row ---
        row = {
            "DATE": ts,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
            "weekday_sin": weekday_sin,
            "weekday_cos": weekday_cos,
            "station_code": station["station_code"],
            # use last known pm10 as lag
            "pm10_lag_1": pm10_lag_1,
            "pm10_lag_2": pm10_lag_2,
        }


        # TODO: Add weather preprocessing here if available
        # e.g. row.update(preprocess_weather(weather_for_ts))

        X_new = pd.DataFrame([row])
        X_new = X_new.reindex(columns=model.feature_names_)

        # --- Predict ---
        y_pred = model.predict(X_new)[0]

        pm10_lag_2 = pm10_lag_1
        pm10_lag_1 = y_pred

        forecast_rows.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%MZ"),
            "pm10_pred": float(y_pred)
        })


    return {
        "case_id": case["case_id"],
        "forecast": forecast_rows
    }

print(cat_model.feature_names_)
