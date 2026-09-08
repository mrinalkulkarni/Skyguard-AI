"""
detector.py — SkyGuard AI

Hybrid anomaly detector using:
1. Rule-based detection
2. Rolling Z-score / MAD
3. Isolation Forest
4. Conservative signal fusion

Owner: Member 1
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest


# ============================================================
# CONFIGURATION
# ============================================================

CONTAMINATION = 0.02
RANDOM_STATE = 42

TEMPERATURE_ROC_THRESHOLD = 10
PRESSURE_ROC_THRESHOLD = 30
HUMIDITY_ROC_THRESHOLD = 20

Z_SCORE_THRESHOLD = 3.0

# Strong evidence threshold
FUSION_THRESHOLD = 0.60


# ============================================================
# 1. RULE-BASED DETECTION
# ============================================================

def rule_based_score(row):
    """
    Detects obvious physical sensor anomalies.
    """

    score = 0.0

    temperature_roc = abs(
        row.get("temperature_c_roc", 0)
    )

    pressure_roc = abs(
        row.get("pressure_hpa_roc", 0)
    )

    humidity_roc = abs(
        row.get("humidity_pct_roc", 0)
    )

    # Sudden temperature change
    if temperature_roc >= TEMPERATURE_ROC_THRESHOLD:
        score = max(score, 1.0)

    # Sudden pressure change
    if pressure_roc >= PRESSURE_ROC_THRESHOLD:
        score = max(score, 1.0)

    # Sudden humidity change
    if humidity_roc >= HUMIDITY_ROC_THRESHOLD:
        score = max(score, 1.0)

    # Missing-value indicators
    if row.get("temperature_c_was_missing", False):
        score = max(score, 0.8)

    if row.get("pressure_hpa_was_missing", False):
        score = max(score, 0.8)

    if row.get("humidity_pct_was_missing", False):
        score = max(score, 0.8)

    return score


# ============================================================
# 2. ROLLING Z-SCORE / MAD
# ============================================================

def rolling_zscore_score(row):
    """
    Detects readings that are unusually different from
    their recent 30-reading behavior.
    """

    scores = []

    sensors = [
        "temperature_c",
        "pressure_hpa",
        "humidity_pct"
    ]

    for sensor in sensors:

        value = row.get(sensor)
        mean = row.get(f"{sensor}_mean_30")
        std = row.get(f"{sensor}_std_30")
        mad = row.get(f"{sensor}_mad_30")

        if pd.isna(value) or pd.isna(mean):
            continue

        # -------------------------
        # Standard deviation signal
        # -------------------------

        if not pd.isna(std) and std > 0:

            z_score = abs(
                (value - mean) / std
            )

            if z_score >= Z_SCORE_THRESHOLD:

                scores.append(
                    min(z_score / 6.0, 1.0)
                )

        # -------------------------
        # MAD signal
        # -------------------------

        if not pd.isna(mad) and mad > 0:

            mad_score = abs(
                value - mean
            ) / (
                1.4826 * mad
            )

            if mad_score >= Z_SCORE_THRESHOLD:

                scores.append(
                    min(mad_score / 6.0, 1.0)
                )

    if not scores:
        return 0.0

    return max(scores)


# ============================================================
# 3. ISOLATION FOREST
# ============================================================

def train_isolation_forest(df):
    """
    Trains Isolation Forest using sensor, temporal,
    and physical-consistency features.
    """

    feature_columns = [

        # Current sensor values
        "temperature_c",
        "pressure_hpa",
        "humidity_pct",

        # Rate of change
        "temperature_c_roc",
        "pressure_hpa_roc",
        "humidity_pct_roc",

        # 5-reading statistics
        "temperature_c_mean_5",
        "temperature_c_std_5",
        "temperature_c_mad_5",

        "pressure_hpa_mean_5",
        "pressure_hpa_std_5",
        "pressure_hpa_mad_5",

        "humidity_pct_mean_5",
        "humidity_pct_std_5",
        "humidity_pct_mad_5",

        # 15-reading means
        "temperature_c_mean_15",
        "pressure_hpa_mean_15",
        "humidity_pct_mean_15",

        # 30-reading means
        "temperature_c_mean_30",
        "pressure_hpa_mean_30",
        "humidity_pct_mean_30",

        # Lag features
        "temperature_c_lag_1",
        "temperature_c_lag_2",
        "temperature_c_lag_3",

        "pressure_hpa_lag_1",
        "pressure_hpa_lag_2",
        "pressure_hpa_lag_3",

        "humidity_pct_lag_1",
        "humidity_pct_lag_2",
        "humidity_pct_lag_3",

        # Physical consistency
        "dewpoint_estimate_c"
    ]

    X = df[feature_columns].copy()

    # Remove infinity
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill warm-up NaN values
    X = X.fillna(
        X.median(numeric_only=True)
    )

    model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=200
    )

    predictions = model.fit_predict(X)

    df = df.copy()

    # Raw ML result
    df["ml_anomaly"] = (
        predictions == -1
    )

    # Isolation Forest decision score
    decision = model.decision_function(X)

    # Convert to anomaly strength
    df["ml_score"] = np.clip(
        0.5 - decision,
        0,
        1
    )

    return df, model


# ============================================================
# 4. HYBRID FUSION
# ============================================================

def fuse_anomaly_signals(df):
    """
    Combines multiple anomaly signals.

    ML alone cannot declare a final anomaly.
    Strong physical or statistical evidence is required.
    """

    df = df.copy()

    # Calculate individual signals
    df["rule_score"] = df.apply(
        rule_based_score,
        axis=1
    )

    df["rolling_score"] = df.apply(
        rolling_zscore_score,
        axis=1
    )

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    df["fusion_score"] = (
        0.50 * df["rule_score"]
        + 0.30 * df["rolling_score"]
        + 0.20 * df["ml_score"]
    )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    # Strong combined evidence
    strong_fusion = (
        df["fusion_score"] >= FUSION_THRESHOLD
    )

    # Very strong physical rule
    strong_rule = (
        df["rule_score"] >= 1.0
    )

    # Strong statistical signal + ML support
    strong_statistics = (
        (df["rolling_score"] >= 0.85)
        & (df["ml_score"] >= 0.45)
    )

    # Final anomaly
    df["is_anomaly"] = (
        strong_fusion
        | strong_rule
        | strong_statistics
    )

    return df


# ============================================================
# 5. COMPLETE DETECTOR
# ============================================================

def detect_anomalies(df):
    """
    Runs the complete hybrid anomaly detector.
    """

    df, model = train_isolation_forest(df)

    df = fuse_anomaly_signals(df)

    return df, model


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from src.simulator import (
        load_clean_data,
        inject_anomalies
    )

    from src.preprocessing import preprocess

    print(
        "Starting hybrid anomaly detection..."
    )

    # Load clean data
    clean = load_clean_data()

    # Inject synthetic anomalies
    stream = inject_anomalies(clean)

    # Preprocess data
    features = preprocess(stream)

    print(
        f"Data prepared: {len(features)} rows"
    )

    # Run detector
    results, model = detect_anomalies(
        features
    )

    print(
        "\nDetection complete!"
    )

    print(
        f"Total rows: {len(results)}"
    )

    print(
        f"ML anomalies: "
        f"{results['ml_anomaly'].sum()}"
    )

    print(
        f"Final anomalies: "
        f"{results['is_anomaly'].sum()}"
    )

    # Signal statistics
    print(
        "\nDetection signal summary:"
    )

    print(
        results[
            [
                "rule_score",
                "rolling_score",
                "ml_score",
                "fusion_score",
                "is_anomaly"
            ]
        ].describe()
    )

    # Display first anomalies
    print(
        "\nFirst detected anomalies:"
    )

    print(
        results[
            results["is_anomaly"]
        ][
            [
                "timestamp",
                "temperature_c",
                "pressure_hpa",
                "humidity_pct",
                "rule_score",
                "rolling_score",
                "ml_score",
                "fusion_score",
                "is_anomaly"
            ]
        ].head(10)
    )