import streamlit as st
import pickle
import numpy as np
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('isolation_forest_model.pkl', 'rb') as file:
    detection_model = pickle.load(file)

st.title("Smart Energy Consumption Prediction and Anomaly Detection")
st.write("Predict electricity consumption and detect anomalies.")

hour = st.slider("Hour", 0, 23, 12)
dayofweek = st.slider("Day of Week (0 = Monday)", 0, 6, 0)
month = st.slider("Month", 1, 12, 1)
dayofyear = st.slider("Day of Year", 1, 366, 1)
weekend = st.selectbox("Weekend", [0, 1])
features = np.array([[hour, dayofweek, month, dayofyear, weekend]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    anomaly = detection_model.predict(features)[0]

    st.success(f"Predicted Energy Consumption: {prediction:.2f} MW")

    if anomaly == -1:
        st.error("This input is detected as an anomaly.")
    else:
        st.info("This input is considered normal.")