"""
app.py — SkyGuard AI

Streamlit dashboard for real-time weather anomaly monitoring.

Owner: Member 2
Status: IN PROGRESS
"""

import sys
import os

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Allow Python to find modules inside src/
# ---------------------------------------------------------

SRC_PATH = os.path.join(
    os.path.dirname(__file__),
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


from pipeline import run_pipeline


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("🛡️ SkyGuard AI")

st.subheader(
    "Intelligent Real-Time Weather Sensor Anomaly Detection"
)

st.markdown(
    """
SkyGuard AI monitors **temperature, pressure, and humidity**
sensor readings and identifies unusual patterns using
machine learning and rule-based analysis.
"""
)


# ---------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------

@st.cache_data(show_spinner="Running SkyGuard AI pipeline...")
def load_pipeline():

    df, _ = run_pipeline()

    return df


df = load_pipeline()


# ---------------------------------------------------------
# Latest reading
# ---------------------------------------------------------

latest = df.iloc[-1]


# ---------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------

st.markdown("## 📊 System Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "🌡️ Temperature",
        f"{latest['temperature_c']:.1f} °C"
    )


with col2:
    st.metric(
        "💨 Pressure",
        f"{latest['pressure_hpa']:.1f} hPa"
    )


with col3:
    st.metric(
        "💧 Humidity",
        f"{latest['humidity_pct']:.1f} %"
    )


with col4:
    st.metric(
        "🚨 Anomalies",
        int(df["ml_anomaly"].sum())
    )


# ---------------------------------------------------------
# Sensor Health
# ---------------------------------------------------------

st.markdown("## ❤️ Sensor Health")

health = latest["sensor_health"]

if health >= 75:
    health_status = "HEALTHY"

elif health >= 50:
    health_status = "WARNING"

elif health >= 25:
    health_status = "POOR"

else:
    health_status = "CRITICAL"


col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Sensor Health Score",
        f"{health:.0f}/100"
    )

with col2:
    st.metric(
        "Health Status",
        health_status
    )


st.progress(
    int(health)
)


# ---------------------------------------------------------
# Weather Trends
# ---------------------------------------------------------

st.markdown("## 📈 Weather Trends")

chart_data = df[
    [
        "timestamp",
        "temperature_c",
        "pressure_hpa",
        "humidity_pct"
    ]
].copy()

chart_data = chart_data.set_index("timestamp")


st.line_chart(
    chart_data[
        [
            "temperature_c",
            "pressure_hpa",
            "humidity_pct"
        ]
    ]
)


# ---------------------------------------------------------
# Anomaly Summary
# ---------------------------------------------------------

st.markdown("## 🚨 Anomaly Summary")

anomalies = df[df["ml_anomaly"] == True].copy()


if len(anomalies) == 0:

    st.success(
        "No anomalies detected."
    )

else:

    st.warning(
        f"{len(anomalies)} anomalous readings detected."
    )


# ---------------------------------------------------------
# Severity distribution
# ---------------------------------------------------------

st.markdown("### Severity Distribution")

severity_counts = (
    df["severity"]
    .value_counts()
    .rename_axis("severity")
    .reset_index(name="count")
)

st.bar_chart(
    severity_counts.set_index("severity")
)


# ---------------------------------------------------------
# Anomaly type distribution
# ---------------------------------------------------------

st.markdown("### Anomaly Types")

anomaly_counts = (
    df[df["anomaly_type"] != "NONE"]["anomaly_type"]
    .value_counts()
    .rename_axis("anomaly_type")
    .reset_index(name="count")
)

if len(anomaly_counts) > 0:

    st.bar_chart(
        anomaly_counts.set_index("anomaly_type")
    )

else:

    st.info("No classified anomalies found.")


# ---------------------------------------------------------
# Recent anomalies
# ---------------------------------------------------------

st.markdown("## 🔍 Recent Anomalies")

if len(anomalies) > 0:

    recent = anomalies.tail(10).copy()

    display_columns = [
        "timestamp",
        "temperature_c",
        "pressure_hpa",
        "humidity_pct",
        "anomaly_type",
        "confidence",
        "severity",
        "sensor_health",
        "corrected_value"
    ]

    st.dataframe(
        recent[display_columns],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No anomalies to display."
    )


# ---------------------------------------------------------
# Latest anomaly explanation
# ---------------------------------------------------------

st.markdown("## 🧠 AI Explanation")

if len(anomalies) > 0:

    latest_anomaly = anomalies.iloc[-1]

    st.write(
        f"**Anomaly Type:** "
        f"{latest_anomaly['anomaly_type']}"
    )

    st.write(
        f"**Confidence:** "
        f"{latest_anomaly['confidence']}%"
    )

    st.write(
        f"**Severity:** "
        f"{latest_anomaly['severity']}"
    )

    st.write(
        f"**Explanation:** "
        f"{latest_anomaly['explanation']}"
    )

    if pd.notna(
        latest_anomaly["corrected_value"]
    ):

        st.write(
            f"**Recommended corrected value:** "
            f"{latest_anomaly['corrected_value']}"
        )

else:

    st.success(
        "No anomaly explanation is currently required."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "SkyGuard AI • Intelligent Weather Sensor "
    "Anomaly Detection System • SIH Demo"
)