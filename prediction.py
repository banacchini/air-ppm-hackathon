import joblib
import pandas as pd
import numpy as np

cat_model = joblib.load("models/catboost_best_model.pkl")


import pandas as pd
import numpy as np

def forecast_with_lag(case, model, horizon=24):
    """
    case: dict containing station info, history, target, weather
    model: trained CatBoost model
    horizon: number of hours to forecast (default: 24)
    """

    # --- Validate station ---
    if not case.get("stations") or "station_code" not in case["stations"][0]:
        raise KeyError("Station info or station_code missing")
    station = case["stations"][0]

    # --- Validate history ---
    history = pd.DataFrame(station.get("history", []))
    if history.empty or "timestamp" not in history or "pm10" not in history:
        raise IndexError("History is missing or invalid")
    if len(history) < 2:
        raise IndexError("Not enough history points for lag features")
    history["timestamp"] = pd.to_datetime(history["timestamp"])

    # Start from the last known PM10
    pm10_lag_1 = history["pm10"].iloc[-1]
    pm10_lag_2 = history["pm10"].iloc[-2]

    # --- Validate target ---
    if "prediction_start_time" not in case.get("target", {}):
        raise KeyError("prediction_start_time missing")
    start_time = pd.to_datetime(case["target"]["prediction_start_time"])

    forecast_rows = []

    for i in range(horizon):
        ts = start_time + pd.Timedelta(hours=i)
        year, month, day, hour = ts.year, ts.month, ts.day, ts.hour
        weekday, day_of_year = ts.weekday(), ts.dayofyear

        # Cyclic encoding
        hour_sin, hour_cos = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
        doy_sin, doy_cos = np.sin(2*np.pi*day_of_year/365.25), np.cos(2*np.pi*day_of_year/365.25)
        weekday_sin, weekday_cos = np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)

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
            "pm10_lag_1": pm10_lag_1,
            "pm10_lag_2": pm10_lag_2
        }

        # TODO: Add weather preprocessing
        # row.update(preprocess_weather(weather_for_ts))
        # z jsona robisz dataFrame -> dataFrame traktujesz weather_preprocessingiem (rozbija zlozone kolumny)
        # po preprocessingu robisz approach >>>>>
        # APPROACH (madrzejszy ale ciezszy) !!! MOZE SIE WYPIERDOLIC - 00-08 - imputujesz avg weather z nocy
        # 9-16 - dzien i 17-23 - wieczor ------->> mega wypierdolka jak dostaniesz jedna godzine xdd
        # if len(weather)
        # merge weather z row (dane o time series)
        #


        # --- Keep only columns that exist in the model ---
        X_new = pd.DataFrame([row])
        X_new = X_new[[col for col in model.feature_names_ if col in X_new.columns]]

        # --- Predict ---
        y_pred = model.predict(X_new)[0]

        # Update lags
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
