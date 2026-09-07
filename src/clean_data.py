"""
clean_data.py — SkyGuard AI
One-time script: converts a raw NOAA ISD-Lite .gz file into data/processed/clean.csv
Owner: Member 1
"""

import pandas as pd
import numpy as np

# --- NOAA file downloaded for this project ---
RAW_FILE = "data/raw/010010-99999-2023.gz"

COLUMNS = [
    "year", "month", "day", "hour",
    "air_temp_c_x10", "dew_point_c_x10", "sea_level_pressure_hpa_x10",
    "wind_dir_deg", "wind_speed_ms_x10", "sky_condition",
    "precip_1h_mm_x10", "precip_6h_mm_x10",
]

df = pd.read_csv(
    RAW_FILE,
    sep=r"\s+",
    header=None,
    names=COLUMNS,
    compression="gzip"
)

# ISD-Lite uses -9999 to mean "missing"
df = df.replace(-9999, np.nan)

df["timestamp"] = pd.to_datetime(
    df[["year", "month", "day", "hour"]]
)

df["temperature_c"] = df["air_temp_c_x10"] / 10.0
df["dew_point_c"] = df["dew_point_c_x10"] / 10.0
df["pressure_hpa"] = df["sea_level_pressure_hpa_x10"] / 10.0

# Calculate Relative Humidity from Temperature + Dew Point
A, B = 17.625, 243.04

def relative_humidity(t, td):
    return 100 * (
        np.exp((A * td) / (B + td))
        / np.exp((A * t) / (B + t))
    )

df["humidity_pct"] = relative_humidity(
    df["temperature_c"],
    df["dew_point_c"]
).clip(0, 100)

clean = (
    df[
        [
            "timestamp",
            "temperature_c",
            "pressure_hpa",
            "humidity_pct"
        ]
    ]
    .dropna(
        subset=[
            "temperature_c",
            "pressure_hpa",
            "humidity_pct"
        ]
    )
    .sort_values("timestamp")
    .reset_index(drop=True)
)

clean.to_csv(
    "data/processed/clean.csv",
    index=False
)

print(f"Saved {len(clean)} clean rows to data/processed/clean.csv")
print(clean.head())
print(clean.describe())