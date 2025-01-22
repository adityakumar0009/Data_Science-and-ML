import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(page_title="Restaurant Review System", layout="wide")

# Load Pre-trained Models and Scalers
scaler = joblib.load("scaler.pkl")
model = joblib.load("ml_model.pkl")

# Page Title and Description
st.markdown(
    """
    <div style="background-color: #ff7f50; padding: 15px; border-radius: 10px;">
        <h1 style="text-align: center; color: white;">🍴 Restaurant Review System</h1>
        <p style="text-align: center; color: white; font-size: 18px;">
        Your Guide to Culinary Experiences: Discover, Rate, and Share Restaurant Reviews!
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# Input Section
col1, col2 = st.columns(2)

with col1:
    average_cost = st.number_input(
        "💰 Average cost for two people:",
        min_value=50,
        max_value=999999,
        value=1000,
        step=200,
    )
    table_booking = st.selectbox("📅 Table Booking Available?", ["Yes", "No"])
    online_delivery = st.selectbox("🚚 Online Delivery Available?", ["Yes", "No"])

with col2:
    price_range = st.selectbox(
        "💲 Price Range (1 = Cheapest, 4 = Most Expensive):", [1, 2, 3, 4]
    )
    predict_button = st.button("🔍 Predict Rating")

st.divider()

# Preprocess Input Data
booking_status = 1 if table_booking == "Yes" else 0
delivery_status = 1 if online_delivery == "Yes" else 0
values = [[average_cost, booking_status, delivery_status, price_range]]
x = scaler.transform(np.array(values))

# Prediction Logic
if predict_button:
    st.snow()
    prediction = model.predict(x)
    
    # Rating Interpretation
    rating_map = {
        "Poor": prediction < 2.5,
        "Average": 2.5 <= prediction < 3.5,
        "Good": 3.5 <= prediction < 4.0,
        "Very Good": 4.0 <= prediction < 4.5,
        "Excellent": prediction >= 4.5,
    }
    
    for rating, condition in rating_map.items():
        if condition:
            st.markdown(
                f"""
                <div style="background-color: #f0f0f5; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: #ff4500;">{rating} ⭐️</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
            break
