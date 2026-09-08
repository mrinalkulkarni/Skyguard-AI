"""
rootcause.py — SkyGuard AI

Classifies the likely root cause of detected sensor anomalies.

The classifier uses:
- Sensor values
- Rate of change (ROC)
- Lag values
- Missing-value indicators
- Anomaly detection results

Owner: Member 1
"""

import pandas as pd


# ---------------------------------------------------------
# Thresholds
# ---------------------------------------------------------

TEMPERATURE_CHANGE_THRESHOLD = 10.0
PRESSURE_SPIKE_THRESHOLD = 30.0
PRESSURE_DRIFT_THRESHOLD = 15.0
HUMIDITY_CHANGE_THRESHOLD = 20.0

PLAUSIBLE_TEMPERATURE_MIN = -90.0
PLAUSIBLE_TEMPERATURE_MAX = 60.0

PLAUSIBLE_PRESSURE_MIN = 800.0
PLAUSIBLE_PRESSURE_MAX = 1100.0

PLAUSIBLE_HUMIDITY_MIN = 0.0
PLAUSIBLE_HUMIDITY_MAX = 100.0

FROZEN_TOLERANCE = 0.001


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def is_missing_sensor_value(row):
    """
    Check whether any core sensor value is missing.
    """

    for column in [
        "temperature_c",
        "pressure_hpa",
        "humidity_pct",
    ]:
        missing_flag = row.get(f"{column}_was_missing", False)

        if bool(missing_flag):
            return True

    return False


def is_temperature_bias(row):
    """
    Detect obviously impossible temperature values.
    """

    temperature = row.get("temperature_c")

    if pd.isna(temperature):
        return False

    return (
        temperature < PLAUSIBLE_TEMPERATURE_MIN
        or temperature > PLAUSIBLE_TEMPERATURE_MAX
    )


def is_pressure_bias(row):
    """
    Detect obviously impossible pressure values.
    """

    pressure = row.get("pressure_hpa")

    if pd.isna(pressure):
        return False

    return (
        pressure < PLAUSIBLE_PRESSURE_MIN
        or pressure > PLAUSIBLE_PRESSURE_MAX
    )


def is_humidity_bias(row):
    """
    Detect impossible humidity values.
    """

    humidity = row.get("humidity_pct")

    if pd.isna(humidity):
        return False

    return (
        humidity < PLAUSIBLE_HUMIDITY_MIN
        or humidity > PLAUSIBLE_HUMIDITY_MAX
    )


def is_frozen_sensor(row, sensor):
    """
    Detect whether a sensor has remained almost exactly unchanged
    across several consecutive readings.
    """

    current = row.get(sensor)
    lag1 = row.get(f"{sensor}_lag_1")
    lag2 = row.get(f"{sensor}_lag_2")
    lag3 = row.get(f"{sensor}_lag_3")

    values = [current, lag1, lag2, lag3]

    if any(pd.isna(value) for value in values):
        return False

    return (
        abs(current - lag1) <= FROZEN_TOLERANCE
        and abs(lag1 - lag2) <= FROZEN_TOLERANCE
        and abs(lag2 - lag3) <= FROZEN_TOLERANCE
    )


# ---------------------------------------------------------
# Root-cause classifier
# ---------------------------------------------------------

def classify_root_cause(row):
    """
    Determine the most likely root cause for one observation.
    """

    temperature = row.get("temperature_c")
    pressure = row.get("pressure_hpa")
    humidity = row.get("humidity_pct")

    temperature_roc = row.get("temperature_c_roc", 0.0)
    pressure_roc = row.get("pressure_hpa_roc", 0.0)
    humidity_roc = row.get("humidity_pct_roc", 0.0)

    if pd.isna(temperature_roc):
        temperature_roc = 0.0

    if pd.isna(pressure_roc):
        pressure_roc = 0.0

    if pd.isna(humidity_roc):
        humidity_roc = 0.0

    # -----------------------------------------------------
    # 1. Missing data / communication failure
    # -----------------------------------------------------

    if is_missing_sensor_value(row):
        return "MISSING_DATA"

    # -----------------------------------------------------
    # 2. Impossible sensor values / sensor bias
    # -----------------------------------------------------

    if (
        is_temperature_bias(row)
        or is_pressure_bias(row)
        or is_humidity_bias(row)
    ):
        return "SENSOR_BIAS"

    # -----------------------------------------------------
    # 3. Temperature spike
    # -----------------------------------------------------

    if temperature_roc >= TEMPERATURE_CHANGE_THRESHOLD:
        return "TEMPERATURE_SPIKE"

    # -----------------------------------------------------
    # 4. Temperature drop
    # -----------------------------------------------------

    if temperature_roc <= -TEMPERATURE_CHANGE_THRESHOLD:
        return "TEMPERATURE_DROP"

    # -----------------------------------------------------
    # 5. Pressure spike
    # -----------------------------------------------------

    if pressure_roc >= PRESSURE_SPIKE_THRESHOLD:
        return "PRESSURE_SPIKE"

    # -----------------------------------------------------
    # 6. Pressure drift
    # -----------------------------------------------------

    if abs(pressure_roc) >= PRESSURE_DRIFT_THRESHOLD:
        return "PRESSURE_DRIFT"

    # -----------------------------------------------------
    # 7. Humidity spike
    # -----------------------------------------------------

    if humidity_roc >= HUMIDITY_CHANGE_THRESHOLD:
        return "HUMIDITY_SPIKE"

    # -----------------------------------------------------
    # 8. Frozen / stuck temperature sensor
    # -----------------------------------------------------

    if is_frozen_sensor(row, "temperature_c"):
        return "TEMPERATURE_FROZEN"

    # -----------------------------------------------------
    # 9. Frozen / stuck pressure sensor
    # -----------------------------------------------------

    if is_frozen_sensor(row, "pressure_hpa"):
        return "PRESSURE_FROZEN"

    # -----------------------------------------------------
    # 10. Frozen / stuck humidity sensor
    # -----------------------------------------------------

    if is_frozen_sensor(row, "humidity_pct"):
        return "HUMIDITY_FROZEN"

    # -----------------------------------------------------
    # 11. Generic sensor frozen condition
    # -----------------------------------------------------

    if (
        is_frozen_sensor(row, "temperature_c")
        or is_frozen_sensor(row, "pressure_hpa")
        or is_frozen_sensor(row, "humidity_pct")
    ):
        return "SENSOR_FROZEN"

    # -----------------------------------------------------
    # 12. Anomaly detected but no specific sensor fault
    # -----------------------------------------------------

    if bool(row.get("is_anomaly", False)):
        return "POSSIBLE_REAL_WEATHER_EVENT"

    # -----------------------------------------------------
    # 13. Normal observation
    # -----------------------------------------------------

    return "NONE"


# ---------------------------------------------------------
# Apply classification to complete dataframe
# ---------------------------------------------------------

def classify_dataframe(df):
    """
    Add anomaly_type column to the dataframe.
    """

    df = df.copy()

    df["anomaly_type"] = df.apply(
        classify_root_cause,
        axis=1
    )

    return df


# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------

if __name__ == "__main__":

    from src.simulator import load_clean_data, inject_anomalies
    from src.preprocessing import preprocess
    from src.detector import detect_anomalies

    print("Loading clean data...")

    clean = load_clean_data()

    print(f"Clean rows: {len(clean)}")

    print("\nInjecting synthetic anomalies...")

    stream = inject_anomalies(clean)

    print(f"Stream rows: {len(stream)}")

    print("\nRunning preprocessing...")

    features = preprocess(stream)

    print(f"Feature rows: {len(features)}")

    print("\nRunning anomaly detection...")

    detected, model = detect_anomalies(features)

    print("Detection complete.")

    print("\nRunning root-cause classification...")

    result = classify_dataframe(detected)

    print("Root-cause classification complete!")

    print("\nAnomaly type counts:")

    print(
        result["anomaly_type"]
        .value_counts()
    )

    print("\nDetected anomalies with root cause:")

    anomaly_columns = [
        "timestamp",
        "temperature_c",
        "pressure_hpa",
        "humidity_pct",
        "is_anomaly",
        "anomaly_type",
    ]

    print(
        result.loc[
            result["is_anomaly"],
            anomaly_columns
        ].head(20).to_string(index=False)
    )