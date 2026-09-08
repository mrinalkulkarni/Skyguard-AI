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

def load_clean_data(file_path=CLEAN_FILE):
    """
    Load cleaned NOAA weather data.
    """
    return pd.read_csv(file_path)

def inject_anomalies(dataframe):
    """
    Inject synthetic anomalies into a copy of the dataframe.
    """

    dataframe = dataframe.copy()

    for index in dataframe.index:

        if index > 0 and index % ANOMALY_INTERVAL == 0:

            anomaly_number = (index // ANOMALY_INTERVAL) % 6

            if anomaly_number == 1:
                dataframe.loc[index, "temperature_c"] += TEMPERATURE_SPIKE

            elif anomaly_number == 2:
                dataframe.loc[index, "temperature_c"] -= TEMPERATURE_DROP

            elif anomaly_number == 3:
                dataframe.loc[index, "pressure_hpa"] += PRESSURE_SPIKE

            elif anomaly_number == 4:
                dataframe.loc[index, "pressure_hpa"] += PRESSURE_DRIFT

            elif anomaly_number == 5:
                dataframe.loc[index, "humidity_pct"] += HUMIDITY_SPIKE
                dataframe.loc[index, "humidity_pct"] = min(
                    dataframe.loc[index, "humidity_pct"],
                    100
                )

            elif anomaly_number == 0:
                dataframe.loc[index, "humidity_pct"] = 50.0

    return dataframe

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
if __name__ == "__main__":
    simulate(df)