"""
correction.py — SkyGuard AI

Suggests a corrected sensor value for detected anomalies.

The original sensor reading is never overwritten.
The recommended value is stored separately in:

    corrected_value

Correction is based on recent rolling sensor history.

Owner: Member 1
"""

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CORRECTION_WINDOW = 5


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def safe_round(value):
    """
    Safely round a numeric value.

    Returns None for missing values.
    """

    if value is None or pd.isna(value):
        return None

    return round(float(value), 2)


def get_previous_mean(row, sensor_column):
    """
    Get the recent rolling mean for a sensor.

    The rolling mean includes the current reading, so we prefer
    a lag-based value when available to avoid using the anomaly
    itself as the correction reference.
    """

    lag_columns = [
        f"{sensor_column}_lag_1",
        f"{sensor_column}_lag_2",
        f"{sensor_column}_lag_3",
    ]

    previous_values = []

    for column in lag_columns:
        value = row.get(column)

        if value is not None and not pd.isna(value):
            previous_values.append(float(value))

    if previous_values:
        return sum(previous_values) / len(previous_values)

    return None


# ---------------------------------------------------------
# Correction logic
# ---------------------------------------------------------

def calculate_corrected_value(row):
    """
    Suggest a corrected value for a detected sensor anomaly.

    Returns:
        float  -> recommended corrected value
        None   -> no correction recommended
    """

    # Never recommend a correction for a normal observation.
    if not bool(row.get("is_anomaly", False)):
        return None

    anomaly_type = str(
        row.get("anomaly_type", "NONE")
    )

    # -----------------------------------------------------
    # Temperature anomalies
    # -----------------------------------------------------

    if anomaly_type in [
        "TEMPERATURE_SPIKE",
        "TEMPERATURE_DROP",
        "TEMPERATURE_DRIFT",
        "TEMPERATURE_FROZEN",
    ]:

        corrected = get_previous_mean(
            row,
            "temperature_c"
        )

        if corrected is not None:
            return safe_round(corrected)

        return None

    # -----------------------------------------------------
    # Pressure anomalies
    # -----------------------------------------------------

    if anomaly_type in [
        "PRESSURE_SPIKE",
        "PRESSURE_DRIFT",
        "PRESSURE_FROZEN",
    ]:

        corrected = get_previous_mean(
            row,
            "pressure_hpa"
        )

        if corrected is not None:
            return safe_round(corrected)

        return None

    # -----------------------------------------------------
    # Humidity anomalies
    # -----------------------------------------------------

    if anomaly_type in [
        "HUMIDITY_SPIKE",
        "HUMIDITY_FROZEN",
    ]:

        corrected = get_previous_mean(
            row,
            "humidity_pct"
        )

        if corrected is not None:

            # Humidity must remain inside 0–100%.
            corrected = max(
                0.0,
                min(100.0, corrected)
            )

            return safe_round(corrected)

        return None

    # -----------------------------------------------------
    # Sensor bias
    # -----------------------------------------------------

    if anomaly_type == "SENSOR_BIAS":

        # Use the most recent previous values from all
        # available sensors only when the affected sensor
        # can be identified from the current values.

        temperature = row.get("temperature_c")
        pressure = row.get("pressure_hpa")
        humidity = row.get("humidity_pct")

        if temperature is not None and not pd.isna(temperature):
            corrected = get_previous_mean(
                row,
                "temperature_c"
            )

            if corrected is not None:
                return safe_round(corrected)

        if pressure is not None and not pd.isna(pressure):
            corrected = get_previous_mean(
                row,
                "pressure_hpa"
            )

            if corrected is not None:
                return safe_round(corrected)

        if humidity is not None and not pd.isna(humidity):
            corrected = get_previous_mean(
                row,
                "humidity_pct"
            )

            if corrected is not None:
                return safe_round(
                    max(0.0, min(100.0, corrected))
                )

        return None

    # -----------------------------------------------------
    # Missing data
    # -----------------------------------------------------

    if anomaly_type == "MISSING_DATA":

        # For missing data, use the most recent available
        # previous reading from each sensor.
        for sensor in [
            "temperature_c",
            "pressure_hpa",
            "humidity_pct",
        ]:

            corrected = get_previous_mean(
                row,
                sensor
            )

            if corrected is not None:

                if sensor == "humidity_pct":
                    corrected = max(
                        0.0,
                        min(100.0, corrected)
                    )

                return safe_round(corrected)

        return None

    # -----------------------------------------------------
    # Uncertain anomalies
    # -----------------------------------------------------

    # We do not automatically correct anomalies that may
    # represent genuine weather events or complex
    # multivariate patterns.

    return None


# ---------------------------------------------------------
# Add corrections to dataframe
# ---------------------------------------------------------

def add_corrections(df):
    """
    Add the recommended corrected value column.
    """

    df = df.copy()

    df["corrected_value"] = df.apply(
        calculate_corrected_value,
        axis=1
    )

    return df


# ---------------------------------------------------------
# Test correction system
# ---------------------------------------------------------

if __name__ == "__main__":

    from src.simulator import (
        load_clean_data,
        inject_anomalies
    )

    from src.preprocessing import preprocess
    from src.detector import detect_anomalies
    from src.rootcause import classify_dataframe
    from src.scoring import add_confidence_scores
    from src.severity import add_severity
    from src.health import add_sensor_health

    print("Starting correction system...")

    # Step 1: Load clean data
    clean = load_clean_data()

    print(
        f"Clean rows: {len(clean)}"
    )

    # Step 2: Inject simulated anomalies
    stream = inject_anomalies(clean)

    print(
        f"Stream rows: {len(stream)}"
    )

    # Step 3: Preprocessing
    features = preprocess(stream)

    print(
        f"Feature rows: {len(features)}"
    )

    # Step 4: Anomaly detection
    detected, model = detect_anomalies(
        features
    )

    print(
        "Detection complete."
    )

    # Step 5: Root cause
    classified = classify_dataframe(
        detected
    )

    print(
        "Root-cause classification complete."
    )

    # Step 6: Confidence
    scored = add_confidence_scores(
        classified
    )

    print(
        "Confidence calculation complete."
    )

    # Step 7: Severity
    severity_data = add_severity(
        scored
    )

    print(
        "Severity calculation complete."
    )

    # Step 8: Sensor health
    health_data = add_sensor_health(
        severity_data
    )

    print(
        "Sensor health calculation complete."
    )

    # Step 9: Recommended correction
    result = add_corrections(
        health_data
    )

    print(
        "Correction calculation complete!"
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    print(
        "\nCorrection column check:"
    )

    if "corrected_value" in result.columns:
        print(
            "corrected_value column: OK"
        )
    else:
        print(
            "corrected_value column: FAILED"
        )

    # Normal rows should never receive corrections.
    normal_rows = result[
        ~result["is_anomaly"]
    ]

    normal_with_correction = normal_rows[
        normal_rows["corrected_value"].notna()
    ]

    if len(normal_with_correction) == 0:
        print(
            "Normal observation correction check: OK"
        )
    else:
        print(
            "Normal observation correction check: FAILED"
        )

    # -----------------------------------------------------
    # Display sample
    # -----------------------------------------------------

    corrected = result[
        result["corrected_value"].notna()
    ]

    print(
        f"\nTotal correction recommendations: "
        f"{len(corrected)}"
    )

    print(
        "\nSample correction recommendations:"
    )

    columns = [
        "timestamp",
        "temperature_c",
        "pressure_hpa",
        "humidity_pct",
        "anomaly_type",
        "confidence",
        "severity",
        "corrected_value",
    ]

    print(
        corrected[
            columns
        ].head(10).to_string(index=False)
    )