"""
preprocessing.py — SkyGuard AI

Cleans and prepares sensor data before anomaly detection.

Owner: Member 1
Status: IN PROGRESS
"""

import pandas as pd


# -----------------------------------------
# Configuration
# -----------------------------------------

CLEAN_FILE = "data/processed/clean.csv"


# -----------------------------------------
# Load data
# -----------------------------------------

def load_data(file_path=CLEAN_FILE):
    """Load cleaned weather sensor data."""

    df = pd.read_csv(file_path)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"Loaded {len(df)} rows")

    return df


# -----------------------------------------
# Handle missing values
# -----------------------------------------

def handle_missing_values(df):
    """Handle missing sensor readings."""

    sensor_columns = [
        "temperature_c",
        "pressure_hpa",
        "humidity_pct"
    ]

    print("\nMissing values before processing:")
    print(df[sensor_columns].isnull().sum())

    # Forward fill missing sensor readings
    df[sensor_columns] = df[sensor_columns].ffill()

    # If missing values are at the beginning,
    # use backward fill
    df[sensor_columns] = df[sensor_columns].bfill()

    print("\nMissing values after processing:")
    print(df[sensor_columns].isnull().sum())

    return df


# -----------------------------------------
# Remove duplicate records
# -----------------------------------------

def remove_duplicates(df):
    """Remove duplicate timestamps."""

    before = len(df)

    df = df.drop_duplicates(subset=["timestamp"])

    after = len(df)

    print(f"\nRemoved duplicates: {before - after}")

    return df


# -----------------------------------------
# Validate sensor ranges
# -----------------------------------------

def validate_sensor_ranges(df):
    """Check that sensor values are physically reasonable."""

    # Temperature range
    df.loc[
        (df["temperature_c"] < -90) |
        (df["temperature_c"] > 60),
        "temperature_c"
    ] = pd.NA

    # Pressure range
    df.loc[
        (df["pressure_hpa"] < 800) |
        (df["pressure_hpa"] > 1100),
        "pressure_hpa"
    ] = pd.NA

    # Humidity range
    df.loc[
        (df["humidity_pct"] < 0) |
        (df["humidity_pct"] > 100),
        "humidity_pct"
    ] = pd.NA

    # Fill values made missing by validation
    sensor_columns = [
        "temperature_c",
        "pressure_hpa",
        "humidity_pct"
    ]

    df[sensor_columns] = df[sensor_columns].ffill().bfill()

    print("\nSensor range validation complete")

    return df


# -----------------------------------------
# Feature engineering
# -----------------------------------------

def create_features(df):
    """Create basic features for anomaly detection."""

    # Change from previous reading
    df["temperature_change"] = df["temperature_c"].diff()

    df["pressure_change"] = df["pressure_hpa"].diff()

    df["humidity_change"] = df["humidity_pct"].diff()

    # Rolling averages
    df["temperature_rolling_mean"] = (
        df["temperature_c"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    df["pressure_rolling_mean"] = (
        df["pressure_hpa"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    df["humidity_rolling_mean"] = (
        df["humidity_pct"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # First row has no previous value
    df = df.fillna(0)

    return df


# -----------------------------------------
# Complete preprocessing pipeline
# -----------------------------------------

def preprocess_data(file_path=CLEAN_FILE):
    """Run the complete preprocessing pipeline."""

    print("\nStarting preprocessing...")

    # Step 1: Load
    df = load_data(file_path)

    # Step 2: Missing values
    df = handle_missing_values(df)

    # Step 3: Remove duplicates
    df = remove_duplicates(df)

    # Step 4: Validate sensor ranges
    df = validate_sensor_ranges(df)

    # Step 5: Create features
    df = create_features(df)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("\nPreprocessing complete!")
    print(f"Final rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")

    return df


# -----------------------------------------
# Run when file is executed directly
# -----------------------------------------

if __name__ == "__main__":

    processed_df = preprocess_data()

    print("\nFirst 5 processed rows:")
    print(processed_df.head())

    print("\nColumns:")
    print(processed_df.columns.tolist())