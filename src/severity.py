"""
severity.py — SkyGuard AI

Converts anomaly confidence into a severity level.

Owner: Member 1
Status: IN PROGRESS
"""

import pandas as pd


# ---------------------------------------------------------
# Severity thresholds
# ---------------------------------------------------------

LOW_THRESHOLD = 1
MEDIUM_THRESHOLD = 25
HIGH_THRESHOLD = 50
CRITICAL_THRESHOLD = 75


# ---------------------------------------------------------
# Severity classification
# ---------------------------------------------------------

def calculate_severity(row):
    """
    Convert anomaly confidence into a severity level.
    """

    confidence = row.get("confidence", 0)

    # Normal reading
    if not row.get("ml_anomaly", False):
        return "NONE"

    if confidence >= CRITICAL_THRESHOLD:
        return "CRITICAL"

    elif confidence >= HIGH_THRESHOLD:
        return "HIGH"

    elif confidence >= MEDIUM_THRESHOLD:
        return "MEDIUM"

    else:
        return "LOW"


# ---------------------------------------------------------
# Add severity to dataframe
# ---------------------------------------------------------

def add_severity(df):
    """
    Add severity classification to the dataframe.
    """

    df = df.copy()

    df["severity"] = df.apply(
        calculate_severity,
        axis=1
    )

    return df


# ---------------------------------------------------------
# Test the severity classifier
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Starting severity classification...")

    from detector import detect_anomalies
    from preprocessing import preprocess_data
    from rootcause import classify_dataframe
    from scoring import add_confidence_scores

    # Step 1: Preprocess
    df = preprocess_data()

    # Step 2: ML anomaly detection
    df, model = detect_anomalies(df)

    # Step 3: Root-cause classification
    df = classify_dataframe(df)

    # Step 4: Confidence scoring
    df = add_confidence_scores(df)

    # Step 5: Severity classification
    df = add_severity(df)

    print("\nSeverity classification complete!")

    print("\nSeverity counts:")
    print(df["severity"].value_counts())

    print("\nSample results:")

    print(
        df[
            [
                "timestamp",
                "ml_anomaly",
                "anomaly_type",
                "confidence",
                "severity"
            ]
        ].head(20)
    )