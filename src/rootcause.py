"""
rootcause.py — SkyGuard AI

Classifies detected sensor anomalies using simple rule-based logic.

Owner: Member 1
Status: IN PROGRESS
"""

import pandas as pd


# ---------------------------------------------------------
# Thresholds
# ---------------------------------------------------------

TEMPERATURE_CHANGE_THRESHOLD = 10
PRESSURE_CHANGE_THRESHOLD = 30
HUMIDITY_CHANGE_THRESHOLD = 20


# ---------------------------------------------------------
# Root-cause classification
# ---------------------------------------------------------

def classify_root_cause(row):
    """
    Identify the most likely cause of an anomaly.
    """

    temperature_change = abs(row.get("temperature_change", 0))
    pressure_change = abs(row.get("pressure_change", 0))
    humidity_change = abs(row.get("humidity_change", 0))

    temperature = row.get("temperature_c", 0)
    pressure = row.get("pressure_hpa", 0)
    humidity = row.get("humidity_pct", 0)

    # Temperature spike
    if temperature_change >= TEMPERATURE_CHANGE_THRESHOLD:
        if row.get("temperature_change", 0) > 0:
            return "TEMPERATURE_SPIKE"
        else:
            return "TEMPERATURE_DROP"

    # Pressure anomaly
    if pressure_change >= PRESSURE_CHANGE_THRESHOLD:
        if row.get("pressure_change", 0) > 0:
            return "PRESSURE_SPIKE"
        else:
            return "PRESSURE_DRIFT"

    # Humidity anomaly
    if humidity_change >= HUMIDITY_CHANGE_THRESHOLD:
        return "HUMIDITY_SPIKE"

    # Frozen/stuck humidity
    if humidity == 50.0:
        return "HUMIDITY_FROZEN"

    # Basic physical-range checks
    if temperature < -90 or temperature > 60:
        return "SENSOR_BIAS"

    if pressure < 800 or pressure > 1100:
        return "SENSOR_BIAS"

    if humidity < 0 or humidity > 100:
        return "SENSOR_BIAS"

    # If ML detected an anomaly but no specific rule matched
    if row.get("ml_anomaly", False):
        return "POSSIBLE_REAL_WEATHER_EVENT"

    return "NONE"


# ---------------------------------------------------------
# Apply classifier to complete dataframe
# ---------------------------------------------------------

def classify_dataframe(df):
    """
    Add root-cause classification to every row.
    """

    df = df.copy()

    df["anomaly_type"] = df.apply(
        classify_root_cause,
        axis=1
    )

    return df


# ---------------------------------------------------------
# Test the classifier
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Starting root-cause classification...")

    from detector import detect_anomalies
    from preprocessing import preprocess_data

    # Load and preprocess data
    df = preprocess_data()

    # Run ML anomaly detection
    df, model = detect_anomalies(df)

    # Classify root causes
    df = classify_dataframe(df)

    print("\nRoot-cause classification complete!")

    print("\nAnomaly type counts:")
    print(df["anomaly_type"].value_counts())

    print("\nSample results:")

    print(
        df[
            [
                "timestamp",
                "temperature_c",
                "pressure_hpa",
                "humidity_pct",
                "ml_anomaly",
                "anomaly_type"
            ]
        ].head(20)
    )