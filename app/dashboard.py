import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Telecom Failure Prediction", page_icon="📡")
st.title("📡 Telecom Failure Prediction")
st.caption("Set network conditions and run a failure-risk prediction.")

signal = st.slider("Signal Strength (dBm)", -100, -50, -80)
temp = st.slider("Temperature (°C)", 20, 50, 30)
humidity = st.slider("Humidity (%)", 20, 100, 50)
load = st.slider("Network Load (%)", 0, 100, 50)

if st.button("Predict"):
    try:
        response = requests.post(
            API_URL,
            json={
                "signal_strength": signal,
                "temperature": temp,
                "humidity": humidity,
                "network_load": load,
            },
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()["prediction"]

        if result == 1:
            st.error("⚠️ High Risk of Failure")
        else:
            st.success("✅ Network Stable")
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        st.info("Make sure FastAPI is running: uvicorn app.main:app --reload")
