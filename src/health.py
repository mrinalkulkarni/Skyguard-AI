"""
health.py — SkyGuard AI

Calculates sensor health scores from recent anomaly evidence.

Health score:
    0   = Poor
    100 = Healthy

The score considers:
- Current anomaly status
- Anomaly severity
- Confidence
- Root-cause type
- Recent anomaly frequency

Owner: Member 1
"""

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

HEALTH_START = 100

RECENT_WINDOW = 24

SEVERITY_PENALTIES = {
    "LOW": 2,
    "MEDIUM": 5,
    "HIGH": 10,
    "CRITICAL": 20,
}


# ---------------------------------------------------------
# Calculate health for one row
# ---------------------------------------------------------

def calculate_health_score(row, recent_anomaly_count=0):
    """
    Calculate the sensor health score for one observation.

    The score starts at 100 and is reduced according to:
    - Current severity
    - Confidence
    - Recent anomaly frequency
    """

    health = HEALTH_START

    is_anomaly = bool(
        row.get("is_anomaly", False)
    )

    severity = str(
        row.get("severity", "NONE")
    )

    confidence = float(
        row.get("confidence", 0) or 0
    )

    # -----------------------------------------------------
    # Current anomaly penalty
    # -----------------------------------------------------

    if is_anomaly:

        severity_penalty = SEVERITY_PENALTIES.get(
            severity,
            0
        )

        health -= severity_penalty

        # Additional penalty for very high confidence.
        if confidence >= 90:
            health -= 5

        elif confidence >= 75:
            health -= 3

    # -----------------------------------------------------
    # Recent anomaly frequency penalty
    # -----------------------------------------------------

    if recent_anomaly_count >= 10:
        health -= 20

    elif recent_anomaly_count >= 5:
        health -= 10

    elif recent_anomaly_count >= 3:
        health -= 5

    elif recent_anomaly_count >= 1:
        health -= 2

    # -----------------------------------------------------
    # Keep score inside 0–100
    # -----------------------------------------------------

    health = max(
        0,
        min(100, health)
    )

    return int(health)


# ---------------------------------------------------------
# Add sensor health to dataframe
# ---------------------------------------------------------

def add_sensor_health(df, window=RECENT_WINDOW):
    """
    Add sensor_health column.

    The recent anomaly count is calculated using a rolling
    window so that health reflects recent sensor behavior.
    """

    df = df.copy()

    # Make sure the dataframe is in chronological order.
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    # Convert anomaly flag to integer for rolling count.
    anomaly_indicator = (
        df["is_anomaly"]
        .astype(int)
    )

    # Count anomalies in the current + previous readings.
    recent_anomaly_count = (
        anomaly_indicator
        .rolling(
            window=window,
            min_periods=1
        )
        .sum()
        .shift(1)
        .fillna(0)
    )

    df["recent_anomaly_count"] = (
        recent_anomaly_count
        .astype(int)
    )

    # Calculate health.
    df["sensor_health"] = df.apply(
        lambda row: calculate_health_score(
            row,
            row["recent_anomaly_count"]
        ),
        axis=1
    )

    return df


# ---------------------------------------------------------
# Health status helper
# ---------------------------------------------------------

def health_status(score):
    """
    Convert numerical health score into a readable status.
    """

    if score >= 90:
        return "HEALTHY"

    if score >= 75:
        return "GOOD"

    if score >= 50:
        return "WARNING"

    return "POOR"


# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------

if __name__ == "__main__":

    from src.simulator import (
        load_clean_data,
        inject_anomalies,
    )

    from src.preprocessing import preprocess

    from src.detector import detect_anomalies

    from src.rootcause import classify_dataframe

    from src.scoring import add_confidence_scores

    from src.severity import add_severity

    print("Loading clean data...")

    clean = load_clean_data()

    print(
        f"Clean rows: {len(clean)}"
    )

    print("\nInjecting anomalies...")

    stream = inject_anomalies(
        clean
    )

    print(
        f"Stream rows: {len(stream)}"
    )

    print("\nRunning preprocessing...")

    features = preprocess(
        stream
    )

    print(
        f"Feature rows: {len(features)}"
    )

    print("\nRunning anomaly detection...")

    detected, model = detect_anomalies(
        features
    )

    print(
        "Detection complete."
    )

    print(
        "\nRunning root-cause classification..."
    )

    classified = classify_dataframe(
        detected
    )

    print(
        "Root-cause classification complete."
    )

    print(
        "\nCalculating confidence..."
    )

    scored = add_confidence_scores(
        classified
    )

    print(
        "Confidence calculation complete."
    )

    print(
        "\nCalculating severity..."
    )

    severity_data = add_severity(
        scored
    )

    print(
        "Severity calculation complete."
    )

    print(
        "\nCalculating sensor health..."
    )

    result = add_sensor_health(
        severity_data
    )

    print(
        "Sensor health calculation complete!"
    )

    # -----------------------------------------------------
    # Health statistics
    # -----------------------------------------------------

    print(
        "\nSensor health statistics:"
    )

    print(
        result["sensor_health"].describe()
    )

    # -----------------------------------------------------
    # Health range validation
    # -----------------------------------------------------

    invalid_health = result[
        (result["sensor_health"] < 0)
        | (result["sensor_health"] > 100)
    ]

    if len(invalid_health) == 0:
        print(
            "\nHealth range check: OK"
        )
    else:
        print(
            "\nHealth range check: FAILED"
        )

    # -----------------------------------------------------
    # Health status distribution
    # -----------------------------------------------------

    result["health_status"] = (
        result["sensor_health"]
        .apply(health_status)
    )

    print(
        "\nHealth status counts:"
    )

    print(
        result["health_status"]
        .value_counts()
    )

    # -----------------------------------------------------
    # Show first anomalies
    # -----------------------------------------------------

    columns_to_show = [
        "timestamp",
        "is_anomaly",
        "anomaly_type",
        "confidence",
        "severity",
        "recent_anomaly_count",
        "sensor_health",
        "health_status",
    ]

    print(
        "\nFirst detected anomalies:"
    )

    print(
        result.loc[
            result["is_anomaly"],
            columns_to_show
        ]
        .head(20)
        .to_string(index=False)
    )