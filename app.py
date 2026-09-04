import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import folium
from streamlit_folium import st_folium

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="AI Flood Prediction & GIS",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 AI-Based Flood Prediction, GIS Mapping & Public Safety")
st.write("AI-powered flood risk prediction and GIS-based safety management system")

# -----------------------------
# SAMPLE TRAINING DATA
# -----------------------------
data = pd.DataFrame({
    "rainfall": [20, 35, 50, 65, 80, 100, 120, 140, 160, 180,
                 200, 220, 240, 260, 280],
    "water_level": [1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.6, 3.0,
                    3.3, 3.6, 4.0, 4.3, 4.7, 5.0, 5.5],
    "temperature": [32, 31, 30, 30, 29, 29, 28, 28, 27, 27,
                    27, 26, 26, 25, 25],
    "risk": [
        "Low", "Low", "Low", "Low", "Low",
        "Medium", "Medium", "Medium", "Medium", "Medium",
        "High", "High", "High", "High", "High"
    ]
})

# -----------------------------
# AI MODEL
# -----------------------------
X = data[["rainfall", "water_level", "temperature"]]
y = data["risk"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# USER INPUT
# -----------------------------
# -----------------------------
# FLOOD PREDICTION
# -----------------------------
st.header("📊 Flood Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=150.0
    )

with col2:
    water_level = st.number_input(
        "Water Level (m)",
        min_value=0.0,
        max_value=10.0,
        value=4.0
    )

with col3:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=28.0
    )

# -----------------------------
# LOCATION SELECTION
# -----------------------------
st.subheader("📍 Select Location")

locations = {
    "Hyderabad": (17.3850, 78.4867),
    "Vijayawada": (16.5062, 80.6480),
    "Visakhapatnam": (17.6868, 83.2185),
    "Warangal": (17.9784, 79.5941),
    "Chennai": (13.0827, 80.2707)
}

selected_location = st.selectbox(
    "Choose a location",
    list(locations.keys())
)

latitude, longitude = locations[selected_location]

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔍 Predict Flood Risk"):

    input_data = pd.DataFrame({
        "rainfall": [rainfall],
        "water_level": [water_level],
        "temperature": [temperature]
    })

    prediction = model.predict(input_data)[0]

    st.session_state["prediction"] = prediction

    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100

    # -----------------------------
    # AI PREDICTION RESULT
    # -----------------------------
    st.subheader("🤖 AI Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📍 Location", selected_location)

    with col2:
        st.metric("🌧️ Rainfall", f"{rainfall} mm")

    with col3:
        st.metric("🌊 Water Level", f"{water_level} m")

    if prediction == "High":
        st.error(
            f"🔴 HIGH FLOOD RISK — Confidence: {confidence:.2f}%"
        )

    elif prediction == "Medium":
        st.warning(
            f"🟠 MEDIUM FLOOD RISK — Confidence: {confidence:.2f}%"
        )

    else:
        st.success(
            f"🟢 LOW FLOOD RISK — Confidence: {confidence:.2f}%"
        )

    # -----------------------------
    # PUBLIC SAFETY
    # -----------------------------
    st.subheader("🚨 Public Safety Management")

    if prediction == "High":
        st.error(f"""
        ⚠️ FLOOD WARNING — {selected_location}

        • Avoid low-lying areas
        • Move to safer locations
        • Avoid flooded roads
        • Follow emergency instructions
        • Contact emergency services if required
        """)

    elif prediction == "Medium":
        st.warning(f"""
        ⚠️ FLOOD WATCH — {selected_location}

        • Monitor water levels
        • Keep emergency supplies ready
        • Avoid unnecessary travel
        • Follow local warnings
        """)

    else:
        st.success(f"""
        ✅ LOW FLOOD RISK — {selected_location}

        • Continue monitoring rainfall
        • Stay updated with local alerts
        """)

# -----------------------------
# GIS MAP
# -----------------------------
st.header("🗺️ GIS Flood Risk Map")

st.write(
    "AI-based GIS visualization of the selected flood-risk location."
)

# Get latest AI prediction
current_risk = st.session_state.get("prediction", "Low")

if current_risk == "High":
    marker_color = "red"
    risk_text = "HIGH RISK"

elif current_risk == "Medium":
    marker_color = "orange"
    risk_text = "MEDIUM RISK"

else:
    marker_color = "green"
    risk_text = "LOW RISK"

# Create map
m = folium.Map(
    location=[latitude, longitude],
    zoom_start=10
)

# AI Risk Marker
folium.Marker(
    [latitude, longitude],
    popup=f"{selected_location} - {risk_text}",
    tooltip=f"AI Prediction: {risk_text}",
    icon=folium.Icon(
        color=marker_color,
        icon="warning-sign"
    )
).add_to(m)

st_folium(
    m,
    width=1100,
    height=500
)

# -----------------------------
# SYSTEM INFORMATION
# -----------------------------
st.header("⚙️ System Components")

st.markdown("""
**Module 1:** AI-Based Flood Prediction and Risk Analysis

**Module 2:** GIS Flood Mapping and Public Safety Management

**Technologies:** Python, Random Forest, Pandas, Scikit-learn, Folium, Streamlit
""")