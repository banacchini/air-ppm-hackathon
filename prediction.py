import joblib
import pandas as pd
import numpy as np
from weather_preprocessing import preprocess_weather

cat_model = joblib.load("models/catboost_best_model.pkl")



def find_weather_for_timestamp(weather_df, target_timestamp):
    """
    Find weather for given timestamp with priority on same hour from different days:
    1. Same hour from nearest days (priority)
    2. Nearest hour if no same hour available  
    3. Average if multiple measurements at same hour
    """
    if weather_df.empty:
        return {}
    
    # Convert to datetime if not already
    weather_df = weather_df.copy()
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
    target_timestamp = pd.to_datetime(target_timestamp)
    
    target_hour = target_timestamp.hour
    
    # Step 1: Try to find same hour from different days
    same_hour_mask = weather_df['DATE'].dt.hour == target_hour
    same_hour_data = weather_df[same_hour_mask]
    
    if not same_hour_data.empty:
        # Found weather data at the same hour - prioritize by date proximity
        target_date = target_timestamp.date()
        date_diffs = same_hour_data['DATE'].dt.date.apply(lambda x: abs((x - target_date).days))
        min_date_diff = date_diffs.min()
        
        # Get all measurements from the closest date(s) with same hour
        closest_same_hour = same_hour_data[date_diffs == min_date_diff]
        
        if len(closest_same_hour) == 1:
            # One measurement at same hour from closest date
            result = closest_same_hour.iloc[0].to_dict()
            result.pop('DATE', None)
            return result
        else:
            # Multiple measurements at same hour - average them
            numeric_cols = closest_same_hour.select_dtypes(include=[np.number]).columns
            result = {}
            
            for col in numeric_cols:
                if col != 'DATE':
                    result[col] = closest_same_hour[col].mean()
            
            # For categorical - take first value
            categorical_cols = closest_same_hour.select_dtypes(exclude=[np.number, 'datetime']).columns
            for col in categorical_cols:
                if col != 'DATE':
                    result[col] = closest_same_hour[col].iloc[0]
                    
            return result
    
    # Step 2: No same hour found - fall back to nearest timestamp (old logic)
    time_diffs = np.abs((weather_df['DATE'] - target_timestamp).dt.total_seconds())
    min_diff = time_diffs.min()
    closest_indices = np.where(time_diffs == min_diff)[0]
    
    if len(closest_indices) == 1:
        # One nearest timestamp
        closest_row = weather_df.iloc[closest_indices[0]]
        result = closest_row.to_dict()
        result.pop('DATE', None)
        return result
    else:
        # Multiple equidistant timestamps - average them
        closest_rows = weather_df.iloc[closest_indices]
        
        numeric_cols = closest_rows.select_dtypes(include=[np.number]).columns
        result = {}
        
        for col in numeric_cols:
            if col != 'DATE':
                result[col] = closest_rows[col].mean()
        
        categorical_cols = closest_rows.select_dtypes(exclude=[np.number, 'datetime']).columns
        for col in categorical_cols:
            if col != 'DATE':
                result[col] = closest_rows[col].iloc[0]
                
        return result

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
        weather_data = case.get("weather", [])
        weather_features = {}

        if weather_data:
            try:
                # Convert to DataFrame
                weather_df = pd.DataFrame(weather_data)
                weather_df.rename(columns={"date": "DATE"}, inplace=True)
                
                # Preprocess weather data
                weather_processed = preprocess_weather(weather_df)
                
                # Find weather for current timestamp
                weather_features = find_weather_for_timestamp(weather_processed, ts)
                
            except Exception as e:
                print(f"Warning: Weather preprocessing failed for {ts}: {e}")

        row.update(weather_features)


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
