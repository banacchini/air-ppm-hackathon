import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def safe_split(series: pd.Series, n_parts: int):
    """Safely split a Series of strings into exactly n_parts columns."""
    return (
        series.fillna("")
        .apply(lambda x: (x.split(",") + [None] * n_parts)[:n_parts])
        .apply(pd.Series)
    )

def preprocess_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
        takes a weather dataframe and preprocess the weather data
        returns dataframe with following columns to match the trained model
        ['DATE', 'wind_speed_raw', 'wind_dir_sin', 'wind_dir_cos',
       'ceiling_coverage', 'visibility_m', 'temperature_C', 'SLP_hpa', 'DEW_C',
       'MA1_main', 'MA1_sec', 'GA1_amt', 'GA1_height', 'GA1_type', 'MD1_m1',
       'MD1_m2']
    """

    df = df.copy()
    df.columns = df.columns.str.upper()

    # --- WND: wind ---
    if "WND" in df.columns:
        wnd_parts = safe_split(df["WND"], 5)
        df["wind_dir_deg"] = pd.to_numeric(wnd_parts[0], errors="coerce")
        df["wind_dir_qc"] = wnd_parts[1]
        df["wind_type"]   = wnd_parts[2]
        df["wind_speed_raw"] = pd.to_numeric(wnd_parts[3], errors="coerce")
        df["wind_speed_qc"] = wnd_parts[4]

        df.loc[df["wind_dir_deg"] == 999, "wind_dir_deg"] = np.nan
        df.loc[df["wind_speed_raw"] == 9999, "wind_speed_raw"] = np.nan
        df.loc[df["wind_type"] == "C", "wind_speed_raw"] = 0
        df.loc[df["wind_type"].isin(["C","V"]), "wind_dir_deg"] = np.nan
        df.loc[df["wind_dir_qc"] != "1", "wind_dir_deg"] = np.nan
        df.loc[df["wind_speed_qc"] != "1", "wind_speed_raw"] = np.nan

        df["wind_dir_sin"] = np.sin(np.radians(df["wind_dir_deg"]))
        df["wind_dir_cos"] = np.cos(np.radians(df["wind_dir_deg"]))
        df = df.drop(columns=["WND", "wind_dir_qc", "wind_type", "wind_speed_qc", "wind_dir_deg"])

    # --- TMP: temperature ---
    if "TMP" in df.columns:
        tmp_parts = safe_split(df["TMP"], 2)
        df["temperature_C"] = pd.to_numeric(tmp_parts[0], errors="coerce") / 10.0
        df["temperature_qlt"] = pd.to_numeric(tmp_parts[1], errors="coerce")
        df.loc[df["temperature_qlt"] != 1, "temperature_C"] = np.nan
        df.drop(columns=["TMP", "temperature_qlt"], inplace=True)

        # --- CIG: ceiling ---
    if "CIG" in df.columns:
        cig_parts = safe_split(df["CIG"], 4)
        df["ceiling_height_ft"] = pd.to_numeric(cig_parts[0], errors="coerce")
        df["ceiling_method"] = cig_parts[1].replace("9", np.nan)
        df["ceiling_quality"] = cig_parts[2].replace("9", np.nan)
        df["ceiling_coverage"] = cig_parts[3].map({"N": 0, "Y": 1, "9": np.nan})
        df.loc[df["ceiling_height_ft"] == 99999, "ceiling_height_ft"] = np.nan
        df.loc[df["ceiling_method"].isna() | df["ceiling_quality"].isna(), "ceiling_height_ft"] = np.nan
        df = df.drop(columns=["CIG", "ceiling_method", "ceiling_quality"])

    # --- VIS: visibility ---
    if "VIS" in df.columns:
        vis_parts = safe_split(df["VIS"], 4)
        df["visibility_m"] = pd.to_numeric(vis_parts[0], errors="coerce")
        df["visibility_var"] = vis_parts[1]
        df["visibility_quality"] = vis_parts[2]
        df["visibility_extra"] = vis_parts[3]
        df.loc[df["visibility_m"].isin([9999, 99999]), "visibility_m"] = np.nan
        df.loc[df["visibility_var"] != "1", "visibility_m"] = np.nan
        df = df.drop(columns=["VIS", "visibility_var", "visibility_quality", "visibility_extra"])

    # --- SLP ---
    if "SLP" in df.columns:
        slp_parts = safe_split(df["SLP"], 2)
        df["SLP_hpa"] = pd.to_numeric(slp_parts[0], errors="coerce") / 10.0
        df["SLP_qlt"] = pd.to_numeric(slp_parts[1], errors="coerce")
        df.loc[df["SLP_qlt"] != 1, "SLP_hpa"] = np.nan
        df.drop(columns=["SLP", "SLP_qlt"], inplace=True)

    # --- DEW ---
    if "DEW" in df.columns:
        dew_parts = safe_split(df["DEW"], 2)
        df["DEW_C"] = pd.to_numeric(dew_parts[0], errors="coerce") / 10.0
        df["DEW_qlt"] = pd.to_numeric(dew_parts[1], errors="coerce")
        df.loc[df["DEW_qlt"] != 1, "DEW_C"] = np.nan
        df.drop(columns=["DEW", "DEW_qlt"], inplace=True)

    # --- MA1 ---
    if "MA1" in df.columns:
        ma1_parts = safe_split(df["MA1"], 4)
        df["MA1_main"] = pd.to_numeric(ma1_parts[0], errors="coerce").replace(99999, np.nan)
        df["MA1_q1"] = ma1_parts[1]
        df["MA1_sec"] = pd.to_numeric(ma1_parts[2], errors="coerce").replace(99999, np.nan)
        df["MA1_q2"] = ma1_parts[3]

        df.loc[df["MA1_q1"] != "1", "MA1_main"] = np.nan
        df.loc[df["MA1_q2"] != "1", "MA1_sec"] = np.nan
        df = df.drop(columns=["MA1", "MA1_q1", "MA1_q2"])

    # --- GA1 ---
    if "GA1" in df.columns:
        ga1_parts = safe_split(df["GA1"], 6)
        df["GA1_amt"] = pd.to_numeric(ga1_parts[0], errors="coerce").replace(99, np.nan)
        df["GA1_q1"] = ga1_parts[1]
        df["GA1_height"] = pd.to_numeric(ga1_parts[2], errors="coerce").replace(99999, np.nan)
        df["GA1_q2"] = ga1_parts[3]
        df["GA1_type"] = pd.to_numeric(ga1_parts[4], errors="coerce").replace(99, np.nan)
        df["GA1_q3"] = ga1_parts[5]

        df.loc[df["GA1_q1"] != "1", "GA1_amt"] = np.nan
        df.loc[df["GA1_q2"] != "1", "GA1_height"] = np.nan
        df.loc[df["GA1_q3"] != "1", "GA1_type"] = np.nan
        df = df.drop(columns=["GA1", "GA1_q1", "GA1_q2", "GA1_q3"])


    # --- MD1 (NEW) ---
    if "MD1" in df.columns:
        md1_parts = safe_split(df["MD1"], 6)
        df["MD1_m1"] = pd.to_numeric(md1_parts[0], errors="coerce")
        df["MD1_q1"] = md1_parts[1]
        df["MD1_m2"] = pd.to_numeric(md1_parts[2], errors="coerce")
        df["MD1_q2"] = md1_parts[3]

        # Replace placeholder codes
        df["MD1_m1"] = df["MD1_m1"].replace(999, np.nan)
        df["MD1_m2"] = df["MD1_m2"].replace(999, np.nan)

        # Keep only reliable measures
        df.loc[df["MD1_q1"] != "1", "MD1_m1"] = np.nan
        df.loc[df["MD1_q2"] != "1", "MD1_m2"] = np.nan

        # Drop original + quality flags
        df = df.drop(columns=["MD1", "MD1_q1", "MD1_q2"])

    # make sure all the weather columns exist
    required_cols = [
        'DATE', 'wind_speed_raw',
       'wind_dir_sin', 'wind_dir_cos', 'ceiling_coverage', 'visibility_m',
       'temperature_C', 'SLP_hpa', 'DEW_C', 'MA1_main', 'MA1_sec', 'GA1_amt',
       'GA1_height', 'GA1_type', 'MD1_m1', 'MD1_m2'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # --- Fill missing categorical columns ---
    for col in ["ceiling_coverage","GA1_type"]:
        if col in df.columns:
            df[col] = df[col].fillna(method="ffill").fillna(0).astype(int)

    # --- Interpolate continuous columns ---
    cont_cols = ["temperature_C","DEW_C","wind_speed_raw","wind_dir_sin","wind_dir_cos",
                 "SLP_hpa","visibility_m","MA1_main","MA1_sec","GA1_amt","GA1_height", "MD1_m1", "MD1_m2"]
    df[cont_cols] = df[cont_cols].interpolate(method="linear", limit_direction="both")



    df = df.reindex(columns=required_cols)

    return df


data = [
    { "date": "2025-01-01T00:00:00", "tmp": "+0050,1", "wnd": "260,1,N,0030,1", "GA1": "abc,1,def,1,ghi,1"}
]

weather = pd.DataFrame(data)
print("Raw input:")
print(weather)

# Run your preprocessing function
processed = preprocess_weather(weather)

print("\nProcessed weather:")
print(processed.head())

print(processed.columns)