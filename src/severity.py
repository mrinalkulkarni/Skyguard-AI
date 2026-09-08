"""
severity.py — SkyGuard AI

Classifies anomaly severity using confidence score and
root-cause type.

Severity levels:
    NONE
    LOW
    MEDIUM
    HIGH
    CRITICAL

Owner: Member 1
"""

import pandas as pd


# ---------------------------------------------------------
# Severity thresholds
# ---------------------------------------------------------

LOW_THRESHOLD = 25
MEDIUM_THRESHOLD = 50
HIGH_THRESHOLD = 75
CRITICAL_THRESHOLD = 90


# ---------------------------------------------------------
# Critical root causes
# ---------------------------------------------------------

CRITICAL_ROOT_CAUSES = {
    "SENSOR_BIAS",
    "MISSING_DATA",
}


# ---------------------------------------------------------
# Calculate severity for one row
# ---------------------------------------------------------

def calculate_severity(row):
    """
    Determine the severity of one observation.
    """

    # Normal observation
    if not bool(row.get("is_anomaly", False)):
        return "NONE"

    confidence = float(
        row.get("confidence", 0) or 0
    )

    anomaly_type = str(
        row.get("anomaly_type", "NONE")
    )

    # -----------------------------------------------------
    # Critical conditions
    # -----------------------------------------------------

    if confidence >= CRITICAL_THRESHOLD:
        return "CRITICAL"

    if anomaly_type in CRITICAL_ROOT_CAUSES and confidence >= HIGH_THRESHOLD:
        return "CRITICAL"

    # -----------------------------------------------------
    # High severity
    # -----------------------------------------------------

    if confidence >= HIGH_THRESHOLD:
        return "HIGH"

    # -----------------------------------------------------
    # Medium severity
    # -----------------------------------------------------

    if confidence >= MEDIUM_THRESHOLD:
        return "MEDIUM"

    # -----------------------------------------------------
    # Low severity
    # -----------------------------------------------------

    if confidence >= LOW_THRESHOLD:
        return "LOW"

    # -----------------------------------------------------
    # Detected anomaly with low confidence
    # -----------------------------------------------------

    return "LOW"


# ---------------------------------------------------------
# Add severity to dataframe
# ---------------------------------------------------------

def add_severity(df):
    """
    Add severity column to the dataframe.
    """

    df = df.copy()

    df["severity"] = df.apply(
        calculate_severity,
        axis=1
    )

    return df


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

    print("Loading clean data...")

    clean = load_clean_data()

    print(f"Clean rows: {len(clean)}")

    print("\nInjecting anomalies...")

    stream = inject_anomalies(clean)

    print(f"Stream rows: {len(stream)}")

    print("\nRunning preprocessing...")

    features = preprocess(stream)

    print(f"Feature rows: {len(features)}")

    print("\nRunning anomaly detection...")

    detected, model = detect_anomalies(features)

    print("Detection complete.")

    print("\nRunning root-cause classification...")

    classified = classify_dataframe(detected)

    print("Root-cause classification complete.")

    print("\nCalculating confidence scores...")

    scored = add_confidence_scores(classified)

    print("Confidence calculation complete.")

    print("\nCalculating severity...")

    result = add_severity(scored)

    print("Severity classification complete!")

    # -----------------------------------------------------
    # Severity distribution
    # -----------------------------------------------------

    print("\nSeverity counts:")

    print(
        result["severity"].value_counts()
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    valid_levels = {
        "NONE",
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    invalid_levels = set(
        result["severity"].unique()
    ) - valid_levels

    if len(invalid_levels) == 0:
        print("\nSeverity level check: OK")
    else:
        print(
            "\nSeverity level check: FAILED"
        )
        print(
            f"Invalid levels: {invalid_levels}"
        )

    # -----------------------------------------------------
    # Check normal observations
    # -----------------------------------------------------

    normal_rows = result[
        ~result["is_anomaly"]
    ]

    normal_with_non_none = normal_rows[
        normal_rows["severity"] != "NONE"
    ]

    if len(normal_with_non_none) == 0:
        print(
            "Normal observation check: OK"
        )
    else:
        print(
            "Normal observation check: FAILED"
        )

    # -----------------------------------------------------
    # Show anomalies
    # -----------------------------------------------------

    columns_to_show = [
        "timestamp",
        "is_anomaly",
        "anomaly_type",
        "confidence",
        "severity",
    ]

    print("\nFirst detected anomalies:")

    print(
        result.loc[
            result["is_anomaly"],
            columns_to_show
        ]
        .head(20)
        .to_string(index=False)
    )