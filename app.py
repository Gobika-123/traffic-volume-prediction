import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from datetime import date

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "traffic_volume_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "model_features.pkl"
DATA_PATH = BASE_DIR / "outputs" / "prepared_traffic_data.csv"
RESULTS_PATH = BASE_DIR / "outputs" / "test_predictions.csv"

st.set_page_config(
    page_title="Traffic Volume Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Traffic Volume Prediction")
st.write(
    "Estimate traffic volume using a machine learning model "
    "trained on historical traffic and weather data."
)

# Verify required files
if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error("Trained model files are missing. Run model_train.py first.")
    st.stop()

# Load saved model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)

model, features = load_model()

# Sidebar inputs
st.sidebar.header("Prediction Inputs")

selected_date = st.sidebar.date_input(
    "Select date",
    value=date(2018, 10, 2),
    min_value=date(2012, 1, 1),
    max_value=date(2035, 12, 31)
)

selected_hour = st.sidebar.slider(
    "Hour of day",
    min_value=0,
    max_value=23,
    value=8
)

temperature = st.sidebar.number_input(
    "Temperature (Kelvin)",
    min_value=0.0,
    max_value=350.0,
    value=288.0
)

rain = st.sidebar.number_input(
    "Rainfall in 1 hour (mm)",
    min_value=0.0,
    max_value=500.0,
    value=0.0
)

snow = st.sidebar.number_input(
    "Snowfall in 1 hour (mm)",
    min_value=0.0,
    max_value=500.0,
    value=0.0
)

clouds = st.sidebar.slider(
    "Cloud coverage (%)",
    min_value=0,
    max_value=100,
    value=40
)

# Predict
if st.sidebar.button("Predict Traffic Volume", type="primary"):
    input_data = pd.DataFrame([{
        "temp": temperature,
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds,
        "hour": selected_hour,
        "day_of_week": selected_date.weekday(),
        "month": selected_date.month,
        "day_of_month": selected_date.day
    }])

    input_data = input_data[features]
    prediction = max(0, float(model.predict(input_data)[0]))

    st.subheader("Prediction Result")
    st.metric(
        "Estimated traffic volume",
        f"{prediction:,.0f} vehicles"
    )

    st.caption(
        "This is a model estimate, not a guaranteed count. "
        "Predictions for dates or conditions outside the training "
        "data may be less reliable."
    )

st.divider()

# Historical trends
st.subheader("📊 Historical Traffic Trends")

if DATA_PATH.exists():
    history = pd.read_csv(DATA_PATH, parse_dates=["date_time"])

    history = history.sort_values("date_time")

    start_date = st.date_input(
        "Historical trend start date",
        value=date(2018, 10, 1),
        key="history_date"
    )

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = start_timestamp + pd.Timedelta(days=1)

    daily = history[
        (history["date_time"] >= start_timestamp) &
        (history["date_time"] < end_timestamp)
    ]

    if not daily.empty:
        chart_data = daily.set_index("date_time")[
            ["traffic_volume"]
        ]

        st.line_chart(chart_data)
        st.write("Historical records:", len(daily))
    else:
        st.info("No records found for this date.")
else:
    st.warning("Prepared historical data was not found.")

# Evaluation results
st.divider()
st.subheader("📈 Model Evaluation")

if RESULTS_PATH.exists() and DATA_PATH.exists():
    actual = pd.read_csv(RESULTS_PATH)

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    mae = mean_absolute_error(
        actual["actual_traffic_volume"],
        actual["predicted_traffic_volume"]
    )

    rmse = mean_squared_error(
        actual["actual_traffic_volume"],
        actual["predicted_traffic_volume"]
    ) ** 0.5

    r2 = r2_score(
        actual["actual_traffic_volume"],
        actual["predicted_traffic_volume"]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f"{mae:.2f}")
    c2.metric("Test RMSE", f"{rmse:.2f}")
    c3.metric("Test R²", f"{r2:.4f}")

    st.caption(
        "Metrics are calculated from the held-out test predictions."
    )

    st.download_button(
        "Download test predictions CSV",
        data=actual.to_csv(index=False).encode("utf-8"),
        file_name="traffic_test_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Run model_train.py to generate evaluation results.")
