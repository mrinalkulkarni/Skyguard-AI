"""
simulator.py — SkyGuard AI

Replays historical weather data row-by-row and injects synthetic anomalies.

Owner: Member 1
Status: IN PROGRESS
"""

import pandas as pd
import time

CLEAN_FILE = "data/processed/clean.csv"

ANOMALY_INTERVAL = 1000
TEMPERATURE_SPIKE = 15
PRESSURE_SPIKE = 50

df = pd.read_csv(CLEAN_FILE)

print(f"Loaded {len(df)} rows")
print(df.head())

def simulate(dataframe, delay=1):
    for index, row in dataframe.iterrows():

        # Copy the row so the original data is not changed
        simulated_row = row.copy()

        # Inject a temperature spike every 1000 rows
        if index > 0 and index % ANOMALY_INTERVAL == 0:
            simulated_row["temperature_c"] += TEMPERATURE_SPIKE
            simulated_row["pressure_hpa"] += PRESSURE_SPIKE
            print("⚠️ SYNTHETIC ANOMALY: Temperature + Pressure spike")

        print(simulated_row.to_dict())
        time.sleep(delay)

simulate(df)