"""
scoring.py — SkyGuard AI

Calculates confidence scores for detected anomalies.

The confidence score combines:
- Rule-based detection
- Rolling/statistical detection
- Machine-learning detection
- Final fusion score
- Root-cause classification

Output:
    confidence: integer from 0 to 100

Owner: Member 1
"""

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CONFIDENCE_MIN = 0
CONFIDENCE_MAX = 100


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def clamp(value, minimum=CONFIDENCE_MIN, maximum=CONFIDENCE_MAX):
    """
    Keep a numeric value inside a fixed range.
    """

    return max(minimum, min(maximum, value))


# ---------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------

def calculate_confidence(row):
    """
    Calculate confidence for one sensor observation.

    The detector already combines multiple signals into
    fusion_score. We use that as the main signal and then
    increase confidence when multiple detection methods agree.
    """

    # Normal observations have zero anomaly confidence.
    if not bool(row.get("is_anomaly", False)):
        return 0

    # Get detector scores.
    rule_score = float(row.get("rule_score", 0.0) or 0.0)
    rolling_score = float(row.get("rolling_score", 0.0) or 0.0)
    ml_score = float(row.get("ml_score", 0.0) or 0.0)
    fusion_score = float(row.get("fusion_score", 0.0) or 0.0)

    # Make sure all scores stay between 0 and 1.
    rule_score = clamp(rule_score, 0.0, 1.0)
    rolling_score = clamp(rolling_score, 0.0, 1.0)
    ml_score = clamp(ml_score, 0.0, 1.0)
    fusion_score = clamp(fusion_score, 0.0, 1.0)

    # -----------------------------------------------------
    # Main confidence component
    # -----------------------------------------------------

    fusion_confidence = fusion_score * 100

    # -----------------------------------------------------
    # Agreement component
    #
    # Count how many detection methods provide a meaningful
    # anomaly signal.
    # -----------------------------------------------------

    supporting_methods = 0

    if rule_score >= 0.50:
        supporting_methods += 1

    if rolling_score >= 0.50:
        supporting_methods += 1

    # ml_score is naturally around a different scale, so
    # 0.45 is used as a meaningful ML anomaly signal.
    if ml_score >= 0.45:
        supporting_methods += 1

    if supporting_methods == 3:
        agreement_bonus = 15

    elif supporting_methods == 2:
        agreement_bonus = 10

    elif supporting_methods == 1:
        agreement_bonus = 5

    else:
        agreement_bonus = 0

    # -----------------------------------------------------
    # Root-cause confirmation
    # -----------------------------------------------------

    anomaly_type = str(
        row.get("anomaly_type", "NONE")
    )

    specific_root_causes = {
        "TEMPERATURE_SPIKE",
        "TEMPERATURE_DROP",
        "TEMPERATURE_DRIFT",
        "PRESSURE_SPIKE",
        "PRESSURE_DRIFT",
        "HUMIDITY_SPIKE",
        "HUMIDITY_FROZEN",
        "TEMPERATURE_FROZEN",
        "PRESSURE_FROZEN",
        "SENSOR_FROZEN",
        "SENSOR_BIAS",
        "MISSING_DATA",
    }

    if anomaly_type in specific_root_causes:
        root_cause_bonus = 5
    else:
        root_cause_bonus = 0

    # -----------------------------------------------------
    # Final confidence
    # -----------------------------------------------------

    confidence = (
        fusion_confidence
        + agreement_bonus
        + root_cause_bonus
    )

    confidence = clamp(
        confidence,
        CONFIDENCE_MIN,
        CONFIDENCE_MAX
    )

    return int(round(confidence))


# ---------------------------------------------------------
# Add confidence column
# ---------------------------------------------------------

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

    result = add_confidence_scores(classified)

    print("Confidence calculation complete!")

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    print("\nConfidence statistics:")

    print(
        result["confidence"].describe()
    )

    print("\nConfidence range:")

    print(
        f"Minimum: {result['confidence'].min()}"
    )

    print(
        f"Maximum: {result['confidence'].max()}"
    )

    # -----------------------------------------------------
    # Check that confidence is valid
    # -----------------------------------------------------

    invalid_confidence = result[
        (result["confidence"] < 0)
        | (result["confidence"] > 100)
    ]

    if len(invalid_confidence) == 0:
        print("\nConfidence range check: OK")
    else:
        print("\nConfidence range check: FAILED")

    # -----------------------------------------------------
    # Show detected anomalies
    # -----------------------------------------------------

    columns_to_show = [
        "timestamp",
        "is_anomaly",
        "anomaly_type",
        "rule_score",
        "rolling_score",
        "ml_score",
        "fusion_score",
        "confidence",
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