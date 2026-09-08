"""
health.py — SkyGuard AI

Calculates the health score of weather sensors.

Owner: Member 1
Status: IN PROGRESS
"""

import pandas as pd


def calculate_health_score(df):
    """
    Calculate sensor health from anomaly history.

    Health score:
    100 = Healthy
    75-99 = Good
    50-74 = Warning
    25-49 = Poor
    0-24 = Critical
    """

    df = df.copy()

    # Start every sensor at 100 health
    health = 100.0

    health_scores = []

    for _, row in df.iterrows():

        # Reduce health when an anomaly occurs
        if row.get("ml_anomaly", False):

            severity = row.get("severity", "LOW")

            if severity == "CRITICAL":
                health -= 10

            elif severity == "HIGH":
                health -= 6

            elif severity == "MEDIUM":
                health -= 3

            else:
                health -= 1

        # Keep health within 0-100
        health = max(0, min(100, health))

        health_scores.append(round(health, 2))

    df["sensor_health"] = health_scores

    return df


def get_health_status(score):
    """
    Convert health score into a readable status.
    """

    if score >= 75:
        return "HEALTHY"

    elif score >= 50:
        return "WARNING"

    elif score >= 25:
        return "POOR"

    else:
        return "CRITICAL"


# ---------------------------------------------------------
# Test the health system
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Starting sensor health calculation...")

    from detector import detect_anomalies
    from preprocessing import preprocess_data
    from rootcause import classify_dataframe
    from scoring import add_confidence_scores
    from severity import add_severity

    # Step 1: Preprocess
    df = preprocess_data()

    # Step 2: Detect anomalies
    df, model = detect_anomalies(df)

    # Step 3: Classify root cause
    df = classify_dataframe(df)

    # Step 4: Calculate confidence
    df = add_confidence_scores(df)

    # Step 5: Calculate severity
    df = add_severity(df)

    # Step 6: Calculate sensor health
    df = calculate_health_score(df)

    print("\nSensor health calculation complete!")

    print(
        f"\nFinal sensor health: "
        f"{df['sensor_health'].iloc[-1]}"
    )

    print(
        f"Health status: "
        f"{get_health_status(df['sensor_health'].iloc[-1])}"
    )

    print("\nSample results:")

    print(
        df[
            [
                "timestamp",
                "ml_anomaly",
                "anomaly_type",
                "confidence",
                "severity",
                "sensor_health"
            ]
        ].head(20)
    )