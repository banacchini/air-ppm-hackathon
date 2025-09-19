import pandas as pd


def clean_weather_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans weather dataframe by interpolating continuous variables
    and filling categorical variables with ffill/mode.
    """

    df = df.copy()

    # Ensure datetime index for interpolation
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').set_index('DATE')

    # Continuous variables → interpolate over time
    cont_cols = [
        "temperature_C", "DEW_C", "wind_speed_raw",
        "wind_dir_sin", "wind_dir_cos", "SLP_hpa", "visibility_m", "GA1_amt", "GA1_height",
        "MA1_main", "MA1_sec", "MD1_m1", "MD1_m2"
    ]
    for col in cont_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="time", limit_direction="both")

    # Categorical/ordinal variables → ffill, then mode if still missing
    cat_cols = ["ceiling_coverage", "GA1_type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
            if df[col].isna().any():
                mode_val = df[col].mode().iloc[0]
                df[col] = df[col].fillna(mode_val)

    # Reset index back to default if needed
    df = df.reset_index()

    return df

dirty_df = pd.read_csv('../data/processed/merged_cleaned_weather_data.csv')

clean = clean_weather_df(dirty_df)

#Check for % of NA values in each column
na_percent = clean.isna().mean() * 100
print("Percentage of NA values in each column:")
print(na_percent)

clean.to_csv("../data/processed/weather_imputed.csv", index=False)