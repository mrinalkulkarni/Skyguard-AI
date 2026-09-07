"""
simulator.py — SkyGuard AI

Replays historical weather data row-by-row and injects synthetic anomalies.

Owner: Member 1
Status: IN PROGRESS
"""

import pandas as pd
import time

CLEAN_FILE = "data/processed/clean.csv"

# Simulator settings
ANOMALY_INTERVAL = 1000
TEMPERATURE_SPIKE = 15
TEMPERATURE_DROP = 15
PRESSURE_SPIKE = 50
PRESSURE_DRIFT = 20
HUMIDITY_SPIKE = 30


# Load cleaned NOAA data
df = pd.read_csv(CLEAN_FILE)

print(f"Loaded {len(df)} rows")
print(df.head())


def simulate(dataframe, delay=1):
    """
    Replay weather data and inject synthetic sensor anomalies.
    """

    for index, row in dataframe.iterrows():

        # Copy the row so the original clean data is never modified
        simulated_row = row.copy()

        anomaly_type = "NONE"

        # -----------------------------------------
        # Synthetic anomaly injection
        # -----------------------------------------

        if index > 0 and index % ANOMALY_INTERVAL == 0:

            anomaly_number = (index // ANOMALY_INTERVAL) % 6

            if anomaly_number == 1:
                # Temperature spike
                simulated_row["temperature_c"] += TEMPERATURE_SPIKE
                anomaly_type = "TEMPERATURE_SPIKE"

            elif anomaly_number == 2:
                # Temperature drop
                simulated_row["temperature_c"] -= TEMPERATURE_DROP
                anomaly_type = "TEMPERATURE_DROP"

            elif anomaly_number == 3:
                # Pressure spike
                simulated_row["pressure_hpa"] += PRESSURE_SPIKE
                anomaly_type = "PRESSURE_SPIKE"

            elif anomaly_number == 4:
                # Pressure drift
                simulated_row["pressure_hpa"] += PRESSURE_DRIFT
                anomaly_type = "PRESSURE_DRIFT"

            elif anomaly_number == 5:
                # Humidity spike
                simulated_row["humidity_pct"] += HUMIDITY_SPIKE
                simulated_row["humidity_pct"] = min(
                    simulated_row["humidity_pct"], 100
                )
                anomaly_type = "HUMIDITY_SPIKE"

            elif anomaly_number == 0:
                # Humidity frozen/stuck
                simulated_row["humidity_pct"] = 50.0
                anomaly_type = "HUMIDITY_FROZEN"

        # -----------------------------------------
        # Output
        # -----------------------------------------

        if anomaly_type != "NONE":
            print(
                f"⚠️ SYNTHETIC ANOMALY: {anomaly_type}"
            )

        print(simulated_row.to_dict())

        time.sleep(delay)


# Start simulation
simulate(df)