"""
pipeline.py — SkyGuard AI

Runs the complete SkyGuard AI anomaly detection pipeline.

Pipeline:
    Data
    ↓
    Simulator
    ↓
    Preprocessing
    ↓
    Anomaly Detection
    ↓
    Root Cause
    ↓
    Confidence
    ↓
    Severity
    ↓
    Sensor Health
    ↓
    Explainability
    ↓
    Correction

Owner: Member 1
"""

import pandas as pd

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
from src.explain import add_explanations
from src.correction import add_corrections


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEFAULT_FILE = "data/processed/clean.csv"


# ---------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------

def run_pipeline(file_path=DEFAULT_FILE):
    """
    Run the complete SkyGuard AI pipeline.

    Returns:
        df     -> final processed dataframe
        model  -> trained Isolation Forest model
    """

    print("\n" + "=" * 60)
    print("SKYGUARD AI — COMPLETE PIPELINE")
    print("=" * 60)

    # -----------------------------------------------------
    # Step 1 — Load clean data
    # -----------------------------------------------------

    print("\n[1/9] Loading clean data...")

    clean = load_clean_data(file_path)

    print(
        f"Clean observations: {len(clean)}"
    )

    # -----------------------------------------------------
    # Step 2 — Simulate streaming anomalies
    # -----------------------------------------------------

    print("\n[2/9] Injecting simulated anomalies...")

    stream = inject_anomalies(clean)

    print(
        f"Stream observations: {len(stream)}"
    )

    # -----------------------------------------------------
    # Step 3 — Preprocessing
    # -----------------------------------------------------

    print("\n[3/9] Preprocessing and feature engineering...")

    features = preprocess(stream)

    print(
        f"Feature observations: {len(features)}"
    )

    # -----------------------------------------------------
    # Step 4 — Anomaly Detection
    # -----------------------------------------------------

    print("\n[4/9] Running anomaly detection...")

    detected, model = detect_anomalies(
        features
    )

    print(
        "Anomaly detection complete."
    )

    # -----------------------------------------------------
    # Step 5 — Root Cause
    # -----------------------------------------------------

    print("\n[5/9] Classifying root causes...")

    classified = classify_dataframe(
        detected
    )

    print(
        "Root-cause classification complete."
    )

    # -----------------------------------------------------
    # Step 6 — Confidence
    # -----------------------------------------------------

    print("\n[6/9] Calculating confidence scores...")

    scored = add_confidence_scores(
        classified
    )

    print(
        "Confidence calculation complete."
    )

    # -----------------------------------------------------
    # Step 7 — Severity
    # -----------------------------------------------------

    print("\n[7/9] Calculating severity...")

    severity_data = add_severity(
        scored
    )

    print(
        "Severity calculation complete."
    )

    # -----------------------------------------------------
    # Step 8 — Sensor Health
    # -----------------------------------------------------

    print("\n[8/9] Calculating sensor health...")

    health_data = add_sensor_health(
        severity_data
    )

    print(
        "Sensor health calculation complete."
    )

    # -----------------------------------------------------
    # Step 9 — Explainability
    # -----------------------------------------------------

    print("\n[9/9] Generating explanations...")

    explained = add_explanations(
        health_data,
        model
    )

    print(
        "Explainability calculation complete."
    )

    # -----------------------------------------------------
    # Recommended correction
    # -----------------------------------------------------

    print("\nCalculating recommended corrections...")

    result = add_corrections(
        explained
    )

    print(
        "Correction calculation complete."
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    anomaly_count = int(
        result["is_anomaly"].sum()
    )

    correction_count = int(
        result["corrected_value"].notna().sum()
    )

    print(
        f"\nTotal readings: {len(result)}"
    )

    print(
        f"Anomalies detected: {anomaly_count}"
    )

    print(
        f"Correction recommendations: {correction_count}"
    )

    print(
        f"Final sensor health: "
        f"{result['sensor_health'].iloc[-1]}"
    )

    print("\nSeverity distribution:")

    print(
        result["severity"].value_counts()
    )

    print("\nAnomaly type distribution:")

    print(
        result["anomaly_type"].value_counts()
    )

    return result, model


# ---------------------------------------------------------
# Final output schema
# ---------------------------------------------------------

def get_final_output(df):
    """
    Return the columns required by the official
    SkyGuard AI output schema.
    """

    output_columns = [
        "timestamp",
        "temperature_c",
        "pressure_hpa",
        "humidity_pct",
        "is_anomaly",
        "anomaly_type",
        "confidence",
        "severity",
        "explanation",
        "sensor_health",
        "corrected_value",
    ]

    return df[output_columns].copy()


# ---------------------------------------------------------
# Test complete pipeline
# ---------------------------------------------------------

if __name__ == "__main__":

    df, model = run_pipeline()

    print("\n" + "=" * 60)
    print("FINAL OUTPUT SCHEMA")
    print("=" * 60)

    final_output = get_final_output(df)

    print(
        "\nFinal output columns:"
    )

    print(
        final_output.columns.tolist()
    )

    print(
        f"\nTotal final columns: "
        f"{len(final_output.columns)}"
    )

    print("\nFinal sample:")

    print(
        final_output.head(10).to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Schema validation
    # -----------------------------------------------------

    expected_columns = [
        "timestamp",
        "temperature_c",
        "pressure_hpa",
        "humidity_pct",
        "is_anomaly",
        "anomaly_type",
        "confidence",
        "severity",
        "explanation",
        "sensor_health",
        "corrected_value",
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in final_output.columns
    ]

    extra_columns = [
        column
        for column in final_output.columns
        if column not in expected_columns
    ]

    if len(missing_columns) == 0:
        print(
            "\nRequired schema check: OK"
        )
    else:
        print(
            "\nRequired schema check: FAILED"
        )

        print(
            f"Missing columns: {missing_columns}"
        )

    if len(extra_columns) == 0:
        print(
            "Extra-column check: OK"
        )
    else:
        print(
            f"Extra columns in final output: "
            f"{extra_columns}"
        )

    # -----------------------------------------------------
    # Basic data validation
    # -----------------------------------------------------

    if len(final_output) == len(df):
        print(
            "Row count preservation check: OK"
        )
    else:
        print(
            "Row count preservation check: FAILED"
        )

    if (
        final_output["confidence"].between(
            0,
            100
        ).all()
    ):
        print(
            "Confidence range check: OK"
        )
    else:
        print(
            "Confidence range check: FAILED"
        )

    if (
        final_output["sensor_health"].between(
            0,
            100
        ).all()
    ):
        print(
            "Sensor health range check: OK"
        )
    else:
        print(
            "Sensor health range check: FAILED"
        )

    print("\n" + "=" * 60)
    print("END-TO-END TEST COMPLETE")
    print("=" * 60)