"""
explain.py — SkyGuard AI

Creates human-readable explanations for detected anomalies.

Uses:
- Root-cause classification
- Sensor values and rate of change
- Isolation Forest + SHAP for model explanation

Owner: Member 1
"""

import numpy as np
import pandas as pd
import shap


MODEL_FEATURES = [
    "temperature_c",
    "pressure_hpa",
    "humidity_pct",

    "temperature_c_roc",
    "pressure_hpa_roc",
    "humidity_pct_roc",

    "temperature_c_mean_5",
    "temperature_c_std_5",
    "temperature_c_mad_5",

    "pressure_hpa_mean_5",
    "pressure_hpa_std_5",
    "pressure_hpa_mad_5",

    "humidity_pct_mean_5",
    "humidity_pct_std_5",
    "humidity_pct_mad_5",

    "temperature_c_mean_15",
    "pressure_hpa_mean_15",
    "humidity_pct_mean_15",

    "temperature_c_mean_30",
    "pressure_hpa_mean_30",
    "humidity_pct_mean_30",

    "temperature_c_lag_1",
    "temperature_c_lag_2",
    "temperature_c_lag_3",

    "pressure_hpa_lag_1",
    "pressure_hpa_lag_2",
    "pressure_hpa_lag_3",

    "humidity_pct_lag_1",
    "humidity_pct_lag_2",
    "humidity_pct_lag_3",

    "dewpoint_estimate_c",
]


FEATURE_LABELS = {
    "temperature_c": "temperature",
    "pressure_hpa": "pressure",
    "humidity_pct": "humidity",

    "temperature_c_roc": "temperature change",
    "pressure_hpa_roc": "pressure change",
    "humidity_pct_roc": "humidity change",

    "temperature_c_mean_5": "5-reading temperature average",
    "temperature_c_std_5": "5-reading temperature variation",
    "temperature_c_mad_5": "5-reading temperature deviation",

    "pressure_hpa_mean_5": "5-reading pressure average",
    "pressure_hpa_std_5": "5-reading pressure variation",
    "pressure_hpa_mad_5": "5-reading pressure deviation",

    "humidity_pct_mean_5": "5-reading humidity average",
    "humidity_pct_std_5": "5-reading humidity variation",
    "humidity_pct_mad_5": "5-reading humidity deviation",

    "temperature_c_mean_15": "15-reading temperature average",
    "pressure_hpa_mean_15": "15-reading pressure average",
    "humidity_pct_mean_15": "15-reading humidity average",

    "temperature_c_mean_30": "30-reading temperature average",
    "pressure_hpa_mean_30": "30-reading pressure average",
    "humidity_pct_mean_30": "30-reading humidity average",

    "temperature_c_lag_1": "previous temperature",
    "temperature_c_lag_2": "temperature two readings ago",
    "temperature_c_lag_3": "temperature three readings ago",

    "pressure_hpa_lag_1": "previous pressure",
    "pressure_hpa_lag_2": "pressure two readings ago",
    "pressure_hpa_lag_3": "pressure three readings ago",

    "humidity_pct_lag_1": "previous humidity",
    "humidity_pct_lag_2": "humidity two readings ago",
    "humidity_pct_lag_3": "humidity three readings ago",

    "dewpoint_estimate_c": "estimated dewpoint",
}


def get_model_features(model, dataframe):
    """
    Get the exact features used by the trained model.
    """

    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = [
            column
            for column in MODEL_FEATURES
            if column in dataframe.columns
        ]

    return features


def prepare_model_input(dataframe, feature_names):
    """
    Prepare data for SHAP.

    Missing or infinite values are replaced safely.
    """

    X = dataframe[feature_names].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X


def calculate_shap_values(model, X):
    """
    Calculate SHAP values for the Isolation Forest model.
    """

    explainer = shap.TreeExplainer(model)

    try:
        shap_values = explainer.shap_values(
            X,
            check_additivity=False
        )
    except TypeError:
        shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0]

    return shap_values


def get_top_shap_feature(shap_row, feature_names):
    """
    Find the feature with the largest absolute SHAP contribution.
    """

    if len(shap_row) == 0:
        return None, 0.0

    index = int(np.argmax(np.abs(shap_row)))

    feature_name = feature_names[index]
    contribution = float(shap_row[index])

    return feature_name, contribution


def feature_label(feature_name):
    """
    Convert technical feature names into readable names.
    """

    return FEATURE_LABELS.get(
        feature_name,
        feature_name.replace("_", " ")
    )


def safe_float(value, default=0.0):
    """
    Safely convert a value to float.
    """

    if value is None or pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_explanation(
    row,
    top_feature=None,
    shap_contribution=0.0
):
    """
    Build a human-readable explanation.
    """

    if not bool(row.get("is_anomaly", False)):
        return "No anomaly detected; the observation is within the expected pattern."

    anomaly_type = str(
        row.get("anomaly_type", "UNKNOWN")
    )

    temperature = safe_float(
        row.get("temperature_c")
    )

    pressure = safe_float(
        row.get("pressure_hpa")
    )

    humidity = safe_float(
        row.get("humidity_pct")
    )

    temperature_roc = safe_float(
        row.get("temperature_c_roc")
    )

    pressure_roc = safe_float(
        row.get("pressure_hpa_roc")
    )

    humidity_roc = safe_float(
        row.get("humidity_pct_roc")
    )

    explanations = {
        "TEMPERATURE_SPIKE":
            f"Temperature increased sharply by {temperature_roc:.1f} °C in one reading.",

        "TEMPERATURE_DROP":
            f"Temperature decreased sharply by {abs(temperature_roc):.1f} °C in one reading.",

        "TEMPERATURE_DRIFT":
            "Temperature shows an unusual sustained change.",

        "PRESSURE_SPIKE":
            f"Pressure increased sharply by {pressure_roc:.1f} hPa in one reading.",

        "PRESSURE_DRIFT":
            f"Pressure changed unusually by {pressure_roc:.1f} hPa.",

        "HUMIDITY_SPIKE":
            f"Humidity increased sharply by {humidity_roc:.1f} percentage points.",

        "HUMIDITY_FROZEN":
            f"Humidity remained almost unchanged at {humidity:.1f}% across consecutive readings.",

        "TEMPERATURE_FROZEN":
            f"Temperature remained almost unchanged at {temperature:.1f} °C across consecutive readings.",

        "PRESSURE_FROZEN":
            f"Pressure remained almost unchanged at {pressure:.1f} hPa across consecutive readings.",

        "SENSOR_FROZEN":
            "A sensor value remained almost unchanged across consecutive readings.",

        "SENSOR_BIAS":
            "The sensor value is outside the configured plausible physical range.",

        "MISSING_DATA":
            "A missing sensor reading was detected.",

        "MULTIVARIATE_INCONSISTENCY":
            "Multiple sensor variables show an unusual combination that is inconsistent with the expected pattern.",

        "POSSIBLE_REAL_WEATHER_EVENT":
            "The hybrid detector found an unusual weather pattern, but no specific sensor fault signature was identified.",

        "UNKNOWN":
            "The observation was detected as anomalous by the hybrid detection system.",
    }

    explanation = explanations.get(
        anomaly_type,
        explanations["UNKNOWN"]
    )

    if top_feature is not None:
        readable_feature = feature_label(top_feature)

        explanation += (
            f" The strongest machine-learning contribution "
            f"came from {readable_feature}."
        )

    return explanation


def add_explanations(df, model):
    """
    Add an explanation column to the dataframe.

    SHAP is calculated only for detected anomalies.
    """

    df = df.copy()

    df["explanation"] = (
        "No anomaly detected; the observation is within the expected pattern."
    )

    anomaly_mask = df["is_anomaly"].astype(bool)

    if anomaly_mask.sum() == 0:
        return df

    anomaly_data = df.loc[anomaly_mask].copy()

    feature_names = get_model_features(
        model,
        df
    )

    if not feature_names:
        for index, row in anomaly_data.iterrows():
            df.loc[index, "explanation"] = build_explanation(row)

        return df

    X = prepare_model_input(
        anomaly_data,
        feature_names
    )

    try:
        shap_values = calculate_shap_values(
            model,
            X
        )

        for position, (index, row) in enumerate(
            anomaly_data.iterrows()
        ):

            shap_row = shap_values[position]

            top_feature, contribution = get_top_shap_feature(
                shap_row,
                feature_names
            )

            df.loc[index, "explanation"] = build_explanation(
                row,
                top_feature,
                contribution
            )

    except Exception as error:
        print(
            "SHAP explanation warning:",
            error
        )

        for index, row in anomaly_data.iterrows():
            df.loc[index, "explanation"] = build_explanation(row)

    return df


if __name__ == "__main__":

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

    print("Loading clean data...")

    clean = load_clean_data()

    print(
        f"Clean rows: {len(clean)}"
    )

    print("\nInjecting anomalies...")

    stream = inject_anomalies(clean)

    print(
        f"Stream rows: {len(stream)}"
    )

    print("\nRunning preprocessing...")

    features = preprocess(stream)

    print(
        f"Feature rows: {len(features)}"
    )

    print("\nRunning anomaly detection...")

    detected, model = detect_anomalies(features)

    print("Detection complete.")

    print("\nRunning root-cause classification...")

    classified = classify_dataframe(
        detected
    )

    print(
        "Root-cause classification complete."
    )

    print("\nCalculating confidence...")

    scored = add_confidence_scores(
        classified
    )

    print(
        "Confidence calculation complete."
    )

    print("\nCalculating severity...")

    severity_data = add_severity(
        scored
    )

    print(
        "Severity calculation complete."
    )

    print("\nCalculating sensor health...")

    health_data = add_sensor_health(
        severity_data
    )

    print(
        "Sensor health calculation complete."
    )

    print("\nGenerating explanations...")

    result = add_explanations(
        health_data,
        model
    )

    print(
        "Explainability calculation complete!"
    )

    print("\nExplanation column check:")

    if "explanation" in result.columns:
        print("Explanation column: OK")
    else:
        print("Explanation column: FAILED")

    print("\nFirst detected anomalies with explanations:")

    columns = [
        "timestamp",
        "anomaly_type",
        "confidence",
        "severity",
        "sensor_health",
        "explanation",
    ]

    print(
        result.loc[
            result["is_anomaly"],
            columns
        ].head(10).to_string(index=False)
    )

    empty_explanations = result[
        result["is_anomaly"]
        & (
            result["explanation"].isna()
            | (result["explanation"].str.strip() == "")
        )
    ]

    if len(empty_explanations) == 0:
        print(
            "\nExplanation validation: OK"
        )
    else:
        print(
            "\nExplanation validation: FAILED"
        )