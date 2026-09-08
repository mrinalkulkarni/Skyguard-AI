"""
preprocessing.py — SkyGuard AI

Cleans a raw/injected stream and engineers features for anomaly detection:
- Missing-value handling
- Rate of change
- Rolling statistics
- Lag features
- Dewpoint consistency

Owner: Member 1
"""

import pandas as pd
import numpy as np


ROLLING_WINDOWS = [5, 15, 30]

CORE_COLUMNS = [
    "temperature_c",
    "pressure_hpa",
    "humidity_pct"
]


def handle_missing(df):
    """
    Forward-fills missing values.

    A separate boolean column is created for each sensor
    so downstream processing knows which values were originally missing.
    """

    df = df.copy()

    for col in CORE_COLUMNS:
        df[f"{col}_was_missing"] = df[col].isna()
        df[col] = df[col].ffill()

    return df


def compute_rate_of_change(df):
    """
    Calculates the change from the previous reading
    for each sensor.
    """

    df = df.copy()

    for col in CORE_COLUMNS:
        df[f"{col}_roc"] = df[col].diff()

    return df


def compute_rolling_stats(df, windows=ROLLING_WINDOWS):
    """
    Calculates rolling mean, standard deviation and MAD
    for each sensor over multiple time windows.
    """

    df = df.copy()

    for col in CORE_COLUMNS:

        for w in windows:

            roll = df[col].rolling(
                window=w,
                min_periods=max(2, w // 2)
            )

            # Rolling mean
            df[f"{col}_mean_{w}"] = roll.mean()

            # Rolling standard deviation
            df[f"{col}_std_{w}"] = roll.std()

            # Rolling Median Absolute Deviation (MAD)
            df[f"{col}_mad_{w}"] = roll.apply(
                lambda x: np.median(
                    np.abs(x - np.median(x))
                ),
                raw=True
            )

    return df


def compute_lag_features(df, lags=(1, 2, 3)):
    """
    Adds previous readings for each sensor.

    This helps the model understand short-term trends.
    """

    df = df.copy()

    for col in CORE_COLUMNS:

        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df


def compute_dewpoint_consistency(df):
    """
    Estimates dew point from current temperature and humidity.

    This provides a physical-consistency feature between
    temperature and humidity.
    """

    df = df.copy()

    A = 17.625
    B = 243.04

    temperature = df["temperature_c"]

    humidity = df["humidity_pct"].clip(1, 100)

    gamma = (
        np.log(humidity / 100)
        + (A * temperature) / (B + temperature)
    )

    df["dewpoint_estimate_c"] = (
        B * gamma
    ) / (
        A - gamma
    )

    return df


def preprocess(df):
    """
    Runs the complete preprocessing pipeline.
    """

    df = handle_missing(df)

    df = compute_rate_of_change(df)

    df = compute_rolling_stats(df)

    df = compute_lag_features(df)

    df = compute_dewpoint_consistency(df)

    return df


if __name__ == "__main__":

    from src.simulator import load_clean_data, inject_anomalies

    # Load clean historical data
    clean = load_clean_data()

    # Inject synthetic anomalies
    stream = inject_anomalies(clean)

    # Generate preprocessing features
    features = preprocess(stream)

    print(
        f"Input rows: {len(stream)} | "
        f"Output rows: {len(features)}"
    )

    print(
        f"Columns produced: {len(features.columns)}"
    )

    print(features.columns.tolist())

    # The first ~30 rows can legitimately contain NaN
    # because of rolling windows and lag features.
    #
    # We therefore check only rows AFTER the warm-up period.

    nan_after_warmup = features.iloc[30:].isna().sum()

    leaking = nan_after_warmup[
        nan_after_warmup > 0
    ]

    if len(leaking) == 0:

        print(
            "No NaN leaks past row 30 — OK"
        )

    else:

        print(
            "NaN leaks found in these columns "
            "after warm-up:"
        )

        print(leaking)