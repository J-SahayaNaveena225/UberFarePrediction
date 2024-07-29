import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from streamlit_calendar import calendar


def load_model():
    return joblib.load('model.joblib')


def main():
    st.set_page_config(page_title="Fare Predictor", page_icon="🚕", layout="wide")

    st.title('🚕 Fare Predictor')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Trip Details")
        date = st.date_input("Select trip date", datetime.now())
        time = st.time_input("Select trip time")

        # Convert date and time to required features
        hour = time.hour
        day = date.day
        month = date.month
        year=date.year
        day_of_week = date.weekday()

    with col2:
        st.subheader("🚗 Ride Information")
        distance = st.number_input('Distance (km)', min_value=0.1, max_value=100.0, value=5.0, step=0.1)
        passenger_count = st.number_input('Number of passengers', min_value=1, max_value=6, value=2, step=1)

    if st.button('Predict Fare', key='predict'):
        with st.spinner('Calculating fare...'):
            model = load_model()
            input_data = pd.DataFrame({
                'hour': [hour],
                'day': [day],
                'month': [month],
                'year': [year],
                'day_of_week': [day_of_week],
                'distance': [distance],
                'passenger_count': [passenger_count]
            })
            prediction = model.predict(input_data)

        st.success(f'💰 The predicted fare is ${prediction[0]:.2f}')

    st.markdown("---")
    st.subheader("📊 How it works")
    st.write("""
    Our fare predictor uses machine learning to estimate trip costs based on:
    - Date and time of the trip
    - Distance to be traveled
    - Number of passengers

    The model is regularly updated with the latest data to ensure accurate predictions.
    """)

    st.sidebar.image(
        "https://cdn-icons-png.flaticon.com/512/3097/3097180.png",
        width=100)
    st.sidebar.title("About")
    st.sidebar.info(
        "This app predicts taxi fares using machine learning. It's designed to help both passengers and drivers estimate trip costs.")


if __name__ == '__main__':
    main()