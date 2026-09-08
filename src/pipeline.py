"""
pipeline.py — SkyGuard AI

Runs the complete SkyGuard AI anomaly detection pipeline.

Owner: Member 1
Status: IN PROGRESS
"""

import pandas as pd

from preprocessing import preprocess_data
from detector import detect_anomalies
from rootcause import classify_dataframe
from scoring import add_confidence_scores
from severity import add_severity
from health import calculate_health_score
from explain import add_explanations
from correction import add_corrections


def run_pipeline(file_path="data/processed/clean.csv"):
    """
    Run the complete SkyGuard AI pipeline.
    """

    print("\n" + "=" * 60)
    print("SKYGUARD AI — COMPLETE PIPELINE")
    print("=" * 60)

    # -----------------------------------------------------
    # Step 1 — Preprocessing
    # -----------------------------------------------------

    print("\n[1/8] Preprocessing data...")

    df = preprocess_data(file_path)

    # -----------------------------------------------------
    # Step 2 — Machine Learning Detection
    # -----------------------------------------------------

    print("\n[2/8] Running Isolation Forest...")

    df, model = detect_anomalies(df)

    # -----------------------------------------------------
    # Step 3 — Root Cause Classification
    # -----------------------------------------------------

    print("\n[3/8] Classifying root causes...")

    df = classify_dataframe(df)

    # -----------------------------------------------------
    # Step 4 — Confidence Scoring
    # -----------------------------------------------------

    print("\n[4/8] Calculating confidence...")

    df = add_confidence_scores(df)

    # -----------------------------------------------------
    # Step 5 — Severity Classification
    # -----------------------------------------------------

    print("\n[5/8] Calculating severity...")

    df = add_severity(df)

    # -----------------------------------------------------
    # Step 6 — Sensor Health
    # -----------------------------------------------------

    print("\n[6/8] Calculating sensor health...")

    df = calculate_health_score(df)

    # -----------------------------------------------------
    # Step 7 — SHAP Explanation
    # -----------------------------------------------------

    print("\n[7/8] Generating SHAP explanations...")

    df = add_explanations(df, model)

    # -----------------------------------------------------
    # Step 8 — Recommended Correction
    # -----------------------------------------------------

    print("\n[8/8] Calculating recommended corrections...")

    df = add_corrections(df)

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    anomaly_count = int(df["ml_anomaly"].sum())

    print(f"\nTotal readings: {len(df)}")
    print(f"Anomalies detected: {anomaly_count}")

    print(
        f"Final sensor health: "
        f"{df['sensor_health'].iloc[-1]}"
    )

    print("\nSeverity distribution:")
    print(df["severity"].value_counts())

    print("\nAnomaly type distribution:")
    print(df["anomaly_type"].value_counts())

    return df, model


# ---------------------------------------------------------
# Test complete pipeline
# ---------------------------------------------------------

if __name__ == "__main__":

    df, model = run_pipeline()

    print("\nFinal output columns:")

    print(df.columns.tolist())

    print("\nFinal sample:")

    print(
        df[
            [
                "timestamp",
                "temperature_c",
                "pressure_hpa",
                "humidity_pct",
                "ml_anomaly",
                "anomaly_type",
                "confidence",
                "severity",
                "sensor_health",
                "explanation",
                "corrected_value"
            ]
        ].head(10)
    )