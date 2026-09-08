"""
explain.py — SkyGuard AI

Generates simple explanations for Isolation Forest anomaly detections
using SHAP feature contributions.

Owner: Member 1
Status: IN PROGRESS
"""

import pandas as pd
import shap


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


def create_shap_explainer(model, X):
    """
    Create a SHAP TreeExplainer for the Isolation Forest model.
    """

    return shap.TreeExplainer(model)


def explain_anomaly(row, model, X):
    """
    Generate a human-readable explanation for one reading.
    """

    # Select the current row
    row_data = row[FEATURE_COLUMNS].to_frame().T

    # Calculate SHAP values
    explainer = create_shap_explainer(model, X)

    shap_values = explainer.shap_values(row_data)

    # SHAP can return different structures depending on the version.
    if isinstance(shap_values, list):
        values = shap_values[0][0]
    else:
        values = shap_values[0]

    # Find the feature with the largest contribution
    contributions = pd.Series(
        values,
        index=FEATURE_COLUMNS
    )

    top_feature = contributions.abs().idxmax()
    top_value = contributions[top_feature]

    # Human-readable feature names
    feature_names = {
        "temperature_c": "temperature",
        "pressure_hpa": "pressure",
        "humidity_pct": "humidity",
        "temperature_change": "temperature change",
        "pressure_change": "pressure change",
        "humidity_change": "humidity change",
        "temperature_rolling_mean": "temperature trend",
        "pressure_rolling_mean": "pressure trend",
        "humidity_rolling_mean": "humidity trend"
    }

    readable_name = feature_names.get(
        top_feature,
        top_feature
    )

    direction = "increased" if top_value > 0 else "decreased"

    explanation = (
        f"Anomaly detected primarily due to "
        f"{readable_name}. "
        f"The feature contribution {direction} the anomaly score."
    )

    return explanation


def add_explanations(df, model):
    """
    Add SHAP-based explanations to anomaly rows.
    """

    df = df.copy()

    X = df[FEATURE_COLUMNS].copy()

    explanations = []

    for _, row in df.iterrows():

        if row.get("ml_anomaly", False):

            try:
                explanation = explain_anomaly(
                    row,
                    model,
                    X
                )

            except Exception:
                explanation = (
                    "Anomaly detected by the machine-learning model "
                    "based on unusual sensor patterns."
                )

        else:
            explanation = "No significant anomaly detected."

        explanations.append(explanation)

    df["explanation"] = explanations

    return df


# ---------------------------------------------------------
# Test SHAP explanation
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Starting SHAP explanation...")

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

    # Step 6: SHAP explanations
    df = add_explanations(df, model)

    print("\nSHAP explanation complete!")

    print("\nSample explanations:")

    print(
        df[
            [
                "timestamp",
                "ml_anomaly",
                "anomaly_type",
                "confidence",
                "severity",
                "explanation"
            ]
        ].head(20)
    )