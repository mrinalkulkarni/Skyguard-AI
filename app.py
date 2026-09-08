"""
app.py — SkyGuard AI

Streamlit dashboard for intelligent weather sensor
anomaly monitoring.

Owner: Member 2
"""

import sys
import os

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Allow Python to find src/
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
# Header
# ---------------------------------------------------------

st.title("🛡️ SkyGuard AI")

st.subheader(
    "Intelligent Real-Time Weather Sensor "
    "Anomaly Detection System"
)

st.markdown(
    """
SkyGuard AI monitors **temperature, pressure, and humidity**
readings from Automatic Weather Stations and identifies
unusual sensor patterns using a hybrid AI detection system.
"""
)


# ---------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------

@st.cache_data(
    show_spinner="Running SkyGuard AI pipeline..."
)
def load_pipeline():

    df, _ = run_pipeline()

    return df


df = load_pipeline()


# ---------------------------------------------------------
# Latest reading
# ---------------------------------------------------------

latest = df.iloc[-1]


# ---------------------------------------------------------
# System Overview
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

    anomaly_count = int(
        df["is_anomaly"].sum()
    )

    st.metric(
        "🚨 Anomalies",
        anomaly_count
    )


# ---------------------------------------------------------
# Sensor Health
# ---------------------------------------------------------

st.markdown("## ❤️ Sensor Health")

health = int(
    latest["sensor_health"]
)


if health >= 90:

    health_status = "HEALTHY"

elif health >= 75:

    health_status = "GOOD"

elif health >= 50:

    health_status = "WARNING"

else:

    health_status = "POOR"


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Sensor Health Score",
        f"{health}/100"
    )


with col2:

    st.metric(
        "Health Status",
        health_status
    )


st.progress(
    health
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


chart_data["timestamp"] = pd.to_datetime(
    chart_data["timestamp"]
)


chart_data = chart_data.set_index(
    "timestamp"
)


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

anomalies = df[
    df["is_anomaly"]
].copy()


if len(anomalies) == 0:

    st.success(
        "✅ No anomalies detected."
    )

else:

    st.warning(
        f"⚠️ {len(anomalies)} anomalous "
        f"readings detected."
    )


# ---------------------------------------------------------
# Severity Distribution
# ---------------------------------------------------------

st.markdown("### Severity Distribution")

severity_counts = (
    df["severity"]
    .value_counts()
    .rename_axis("severity")
    .reset_index(name="count")
)


st.bar_chart(
    severity_counts.set_index(
        "severity"
    )
)


# ---------------------------------------------------------
# Anomaly Type Distribution
# ---------------------------------------------------------

st.markdown("### Anomaly Types")

anomaly_counts = (
    df[
        df["anomaly_type"] != "NONE"
    ]["anomaly_type"]
    .value_counts()
    .rename_axis("anomaly_type")
    .reset_index(name="count")
)


if len(anomaly_counts) > 0:

    st.bar_chart(
        anomaly_counts.set_index(
            "anomaly_type"
        )
    )

else:

    st.info(
        "No classified anomalies found."
    )


# ---------------------------------------------------------
# Recent Anomalies
# ---------------------------------------------------------

st.markdown("## 🔍 Recent Anomalies")


if len(anomalies) > 0:

    recent = anomalies.tail(
        10
    ).copy()


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
        recent[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )


else:

    st.info(
        "No anomalies to display."
    )


# ---------------------------------------------------------
# AI Explanation
# ---------------------------------------------------------

st.markdown("## 🧠 AI Explanation")


if len(anomalies) > 0:

    latest_anomaly = anomalies.iloc[-1]


    st.info(
        f"**{latest_anomaly['anomaly_type']}**"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Confidence",
            f"{latest_anomaly['confidence']}%"
        )


    with col2:

        st.metric(
            "Severity",
            latest_anomaly["severity"]
        )


    with col3:

        st.metric(
            "Sensor Health",
            f"{latest_anomaly['sensor_health']}/100"
        )


    st.markdown(
        "**Why was this detected?**"
    )


    st.write(
        latest_anomaly["explanation"]
    )


    # -----------------------------------------------------
    # Recommended correction
    # -----------------------------------------------------

    corrected_value = (
        latest_anomaly["corrected_value"]
    )


    if pd.notna(corrected_value):

        st.markdown(
            "**🔧 Recommended Correction**"
        )


        st.success(
            f"Recommended sensor value: "
            f"**{corrected_value:.2f}**"
        )


    else:

        st.info(
            "No automatic correction is recommended "
            "for this anomaly."
        )


else:

    st.success(
        "No anomaly explanation is currently required."
    )


# ---------------------------------------------------------
# Detailed anomaly table
# ---------------------------------------------------------

st.markdown("## 📋 Detection Details")


detail_columns = [
    "timestamp",
    "temperature_c",
    "pressure_hpa",
    "humidity_pct",
    "is_anomaly",
    "anomaly_type",
    "confidence",
    "severity",
    "sensor_health",
    "corrected_value"
]


st.dataframe(
    df[detail_columns].tail(50),
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "SkyGuard AI • Intelligent Weather Sensor "
    "Anomaly Detection System • SIH Demo"
)