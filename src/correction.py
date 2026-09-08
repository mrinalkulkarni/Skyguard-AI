"""
correction.py — SkyGuard AI

Suggests a corrected sensor value for detected anomalies
using the rolling mean of the sensor.

Owner: Member 1
Status: IN PROGRESS
"""

import pandas as pd


# ---------------------------------------------------------
# Correction logic
# ---------------------------------------------------------

def calculate_corrected_value(row):
    """
    Suggest a corrected value using the rolling mean.

    Returns None when the reading is normal.
    """

    if not row.get("ml_anomaly", False):
        return None

    anomaly_type = row.get("anomaly_type", "NONE")

    # Temperature anomalies
    if anomaly_type in [
        "TEMPERATURE_SPIKE",
        "TEMPERATURE_DROP"
    ]:
        return round(
            row.get("temperature_rolling_mean", row["temperature_c"]),
            2
        )

    # Pressure anomalies
    if anomaly_type in [
        "PRESSURE_SPIKE",
        "PRESSURE_DRIFT"
    ]:
        return round(
            row.get("pressure_rolling_mean", row["pressure_hpa"]),
            2
        )

    # Humidity anomalies
    if anomaly_type in [
        "HUMIDITY_SPIKE",
        "HUMIDITY_FROZEN"
    ]:
        return round(
            row.get("humidity_rolling_mean", row["humidity_pct"]),
            2
        )

    # For uncertain anomalies, don't automatically change the value.
    return None


def add_corrections(df):
    """
    Add recommended corrected values to the dataframe.
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

    print("Starting correction system...")

    from detector import detect_anomalies
    from preprocessing import preprocess_data
    from rootcause import classify_dataframe
    from scoring import add_confidence_scores
    from severity import add_severity

    # Step 1: Preprocess
    df = preprocess_data()

    # Step 2: Detect anomalies
    df, model = detect_anomalies(df)

    # Step 3: Root cause
    df = classify_dataframe(df)

    # Step 4: Confidence
    df = add_confidence_scores(df)

    # Step 5: Severity
    df = add_severity(df)

    # Step 6: Recommended correction
    df = add_corrections(df)

    print("\nCorrection system complete!")

    print("\nRows with recommended corrections:")

    corrected = df[df["corrected_value"].notna()]

    print(f"Total corrected recommendations: {len(corrected)}")

    print("\nSample results:")

    print(
        corrected[
            [
                "timestamp",
                "temperature_c",
                "pressure_hpa",
                "humidity_pct",
                "anomaly_type",
                "confidence",
                "severity",
                "corrected_value"
            ]
        ].head(10)
    )