"""
scoring.py — SkyGuard AI

Calculates anomaly confidence scores using:
- ML anomaly detection
- Anomaly strength
- Sensor change magnitude

Owner: Member 1
Status: IN PROGRESS
"""

import pandas as pd


def calculate_confidence(row):
    """
    Calculate a confidence score from 0 to 100.
    """

    # Normal reading
    if not row.get("ml_anomaly", False):
        return 0

    # Start with ML detection confidence
    ml_score = row.get("anomaly_score", 0)

    # Isolation Forest gives lower scores to stronger anomalies.
    # Convert the score into a simple 0-100 confidence value.
    ml_confidence = max(0, min(100, (0.5 - ml_score) * 100))

    # Calculate sensor-change strength
    temperature_change = abs(row.get("temperature_change", 0))
    pressure_change = abs(row.get("pressure_change", 0))
    humidity_change = abs(row.get("humidity_change", 0))

    change_strength = max(
        temperature_change / 20,
        pressure_change / 50,
        humidity_change / 40
    )

    change_confidence = min(100, change_strength * 100)

    # Combine ML confidence and sensor-change confidence
    confidence = (
        0.7 * ml_confidence +
        0.3 * change_confidence
    )

    return int(max(0, min(100, confidence)))


def add_confidence_scores(df):
    """
    Add confidence scores to the dataframe.
    """

    df = df.copy()

    df["confidence"] = df.apply(
        calculate_confidence,
        axis=1
    )

    return df


if __name__ == "__main__":

    print("Starting anomaly scoring...")

    from detector import detect_anomalies
    from preprocessing import preprocess_data
    from rootcause import classify_dataframe

    # Step 1: Preprocess data
    df = preprocess_data()

    # Step 2: Detect anomalies
    df, model = detect_anomalies(df)

    # Step 3: Classify root cause
    df = classify_dataframe(df)

    # Step 4: Calculate confidence
    df = add_confidence_scores(df)

    print("\nAnomaly scoring complete!")

    print("\nConfidence statistics:")
    print(df["confidence"].describe())

    print("\nSample results:")

    print(
        df[
            [
                "timestamp",
                "ml_anomaly",
                "anomaly_type",
                "anomaly_score",
                "confidence"
            ]
        ].head(20)
    )