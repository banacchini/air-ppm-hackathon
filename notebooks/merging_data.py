import pandas as pd
import numpy as np
from datetime import datetime

# Read data from CSV file
input_file = 'data/processed/weather_pre_algorithm.csv'
df = pd.read_csv(input_file)

df['DATE'] = pd.to_datetime(df['DATE'])

# Step 1: Keep only measurements at full hours (minutes = 0)
df_hourly = df[df['DATE'].dt.minute == 0].copy()

print(f"Data before filtering: {len(df)} rows")
print(f"Data after filtering (full hours only): {len(df_hourly)} rows")


# Step 2: Group by hour (without considering REPORT_TYPE yet)
df_hourly['hour'] = df_hourly['DATE'].dt.floor('H')

# Step 3: Function to merge weather data
def merge_weather_data(group):
    """Function to merge FM-12 and FM-15 data for the same hour"""
    
    if len(group) == 1:
        # If there's only one measurement, return it as is
        return group.iloc[0]
    
    # List of numeric columns to average
    numeric_columns = [
        'wind_speed_raw', 'wind_dir_sin', 'wind_dir_cos', 'ceiling_height_ft', 
        'ceiling_coverage', 'visibility_m', 'temperature_C', 'SLP_hpa', 'DEW_C',
        'MA1_main', 'MA1_sec', 'GA1_amt', 'GA1_height', 'GA2_amt', 'GA2_height',
        'MD1_m1', 'MD1_m2', 'MW1_val'
    ]
    
    # Create new row using first observation as base
    merged_row = group.iloc[0].copy()
    
    # For numeric columns - calculate arithmetic mean
    for col in numeric_columns:
        if col in group.columns:
            # Ignore NaN values when calculating mean
            values = group[col].dropna()
            if len(values) > 0:
                merged_row[col] = values.mean()
            else:
                merged_row[col] = np.nan
    
    # For REPORT_TYPE - combine names
    report_types = group['REPORT_TYPE'].unique()
    merged_row['REPORT_TYPE'] = '+'.join(sorted(report_types))
    
    # For text/categorical columns - take first non-null value
    text_columns = ['GA1_type', 'GA2_type']
    for col in text_columns:
        if col in group.columns:
            non_null_values = group[col].dropna()
            if len(non_null_values) > 0:
                merged_row[col] = non_null_values.iloc[0]
    
    return merged_row

# Step 4: Group by hour and merge data
grouped = df_hourly.groupby('hour')
merged_data = []

for hour, group in grouped:
    merged_row = merge_weather_data(group)
    merged_data.append(merged_row)

# Create new DataFrame with processed data
df_final = pd.DataFrame(merged_data)

# Remove helper column 'hour'
df_final = df_final.drop('hour', axis=1)

print(f"\nData after merging: {len(df_final)} rows")

# Display sample results
print("\nFirst few rows of processed dataset:")
print(df_final.head())

# Save to CSV file
output_file = 'processed_weather_data_merged.csv'
df_final.to_csv(output_file, index=False)
print(f"\nData saved to file: {output_file}")

# Check which hours had merged data
print("\nMerging summary:")
for i, row in df_final.iterrows():
    if '+' in str(row['REPORT_TYPE']):
        print(f"Hour {row['DATE']}: merged data {row['REPORT_TYPE']}")

# Display detailed statistics
print(f"\nProcessing statistics:")
print(f"- Original rows: {len(df)}")
print(f"- Rows with full hours: {len(df_hourly)}")
print(f"- Final processed rows: {len(df_final)}")
print(f"- Rows removed (30-minute intervals): {len(df) - len(df_hourly)}")
print(f"- Hours with merged FM-12+FM-15 data: {sum(1 for _, row in df_final.iterrows() if '+' in str(row['REPORT_TYPE']))}")