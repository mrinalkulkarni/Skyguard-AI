"""
detector.py — SkyGuard AI

Detects anomalies in weather sensor data using Isolation Forest.

Owner: Member 1
Status: IN PROGRESS
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

from preprocessing import preprocess_data


# -----------------------------------------
# Configuration
# -----------------------------------------

CONTAMINATION = 0.02
RANDOM_STATE = 42


# -----------------------------------------
# Feature columns
# -----------------------------------------

FEATURE_COLUMNS = [
    "temperature_c",
    "pressure_hpa",
    "humidity_pct",
    "temperature_change",
    "pressure_change",
    "humidity_change",
    "temperature_rolling_mean",
    "pressure_rolling_mean",
    "humidity_rolling_mean"
]


# -----------------------------------------
# Detect anomalies
# -----------------------------------------

def detect_anomalies(df):
    """
    Detect unusual weather sensor readings
    using Isolation Forest.
    """

    # Select features
    X = df[FEATURE_COLUMNS].copy()

    # Create Isolation Forest model
    model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE
    )

    # Train model and predict
    predictions = model.fit_predict(X)

    # Isolation Forest:
    #  1  = normal
    # -1  = anomaly

    df["ml_anomaly"] = predictions == -1

    # Anomaly score
    df["anomaly_score"] = model.decision_function(X)

    return df, model


# -----------------------------------------
# Main execution
# -----------------------------------------

if __name__ == "__main__":

    print("Starting anomaly detection...")

    # Run preprocessing first
    df = preprocess_data()

    print(f"\nData ready for ML: {len(df)} rows")

    # Detect anomalies
    df, model = detect_anomalies(df)

    # Count anomalies
    anomaly_count = df["ml_anomaly"].sum()

    print(f"\nAnomalies detected: {anomaly_count}")

    print("\nFirst 10 detection results:")

    print(
        df[
            [
                "timestamp",
                "temperature_c",
                "pressure_hpa",
                "humidity_pct",
                "ml_anomaly",
                "anomaly_score"
            ]
        ].head(10)
    )