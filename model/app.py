import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Food Delivery Time Predictor Model", layout="centered")

MODEL_PATH = "model/xgb_best.pkl"
SCALER_PATH = "model/standard_scaler.pkl"

FEATURE_ORDER = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Road_traffic_density",
    "Vehicle_condition",
    "multiple_deliveries",
    "distance_km",
    "order_hour",
    "Type_of_order_Drinks",
    "Type_of_order_Meal",
    "Type_of_order_Snack",
    "Type_of_vehicle_electric_scooter",
    "Type_of_vehicle_motorcycle",
    "Type_of_vehicle_scooter",
    "Festival_Yes",
    "City_Semi-Urban",
    "City_Urban",
    "Weatherconditions_Fog",
    "Weatherconditions_Sandstorms",
    "Weatherconditions_Stormy",
    "Weatherconditions_Sunny",
    "Weatherconditions_Windy",
]

# Ordinal mapping for Road_traffic_density (single numeric column, not one-hot).
# NOTE: confirm this matches the mapping used at training time.
TRAFFIC_MAP = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}

# Categories for one-hot columns (base/dropped category noted per group)
ORDER_TYPES = ["Buffet", "Drinks", "Meal", "Snack"]          # base: Buffet
VEHICLE_TYPES = ["bicycle", "electric_scooter", "motorcycle", "scooter"]  # base: bicycle
CITY_TYPES = ["Metropolitian", "Semi-Urban", "Urban"]        # base: Metropolitian
WEATHER_TYPES = ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"]  # base: Cloudy


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def build_feature_row(age, ratings, traffic, vehicle_cond, multi_deliv,
                       distance, order_hour, order_type, vehicle_type,
                       festival, city, weather):
    row = {col: 0.0 for col in FEATURE_ORDER}

    row["Delivery_person_Age"] = age
    row["Delivery_person_Ratings"] = ratings
    row["Road_traffic_density"] = TRAFFIC_MAP[traffic]
    row["Vehicle_condition"] = vehicle_cond
    row["multiple_deliveries"] = multi_deliv
    row["distance_km"] = distance
    row["order_hour"] = order_hour

    if order_type != "Buffet":
        row[f"Type_of_order_{order_type}"] = 1.0

    if vehicle_type != "bicycle":
        row[f"Type_of_vehicle_{vehicle_type}"] = 1.0

    if festival == "Yes":
        row["Festival_Yes"] = 1.0

    if city != "Metropolitian":
        row[f"City_{city}"] = 1.0

    if weather != "Cloudy":
        row[f"Weatherconditions_{weather}"] = 1.0

    return pd.DataFrame([row], columns=FEATURE_ORDER)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("Food Delivery Time Predictor")
st.caption("Predicts estimated delivery time (in minutes) using an XGBoost regression model.")

model, scaler = load_artifacts()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Delivery person age", min_value=15, max_value=65, value=30)
    ratings = st.slider("Delivery person rating", min_value=1.0, max_value=5.0, value=4.6, step=0.1)
    traffic = st.selectbox("Road traffic density", list(TRAFFIC_MAP.keys()), index=1)
    vehicle_cond = st.slider("Vehicle condition (0=poor, 3=excellent)", 0, 3, 1)
    multi_deliv = st.slider("Multiple deliveries (this trip)", 0, 3, 0)
    distance = st.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)

with col2:
    order_hour = st.slider("Order hour (24h)", 0, 23, 18)
    order_type = st.selectbox("Type of order", ORDER_TYPES)
    vehicle_type = st.selectbox("Type of vehicle", VEHICLE_TYPES)
    festival = st.selectbox("Festival day?", ["No", "Yes"])
    city = st.selectbox("City type", CITY_TYPES)
    weather = st.selectbox("Weather conditions", WEATHER_TYPES)

st.divider()

if st.button("Predict delivery time", type="primary", use_container_width=True):
    features = build_feature_row(
        age, ratings, traffic, vehicle_cond, multi_deliv,
        distance, order_hour, order_type, vehicle_type,
        festival, city, weather,
    )

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    st.success(f"### Estimated delivery time: **{prediction:.1f} minutes**")

    with st.expander("See feature vector sent to the model"):
        st.dataframe(features.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "⚠️ Road_traffic_density is assumed encoded as Low=0, Medium=1, High=2, Jam=3. "
    "If your training notebook used a different mapping, update TRAFFIC_MAP in app.py."
)
