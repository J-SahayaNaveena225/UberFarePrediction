import streamlit as st
import joblib
import pandas as pd

def load_model():
    return joblib.load('fare_model.joblib')

def main():
    st.title('Taxi Fare Predictor')
    
    hour = st.slider('Hour of the day', 0, 23, 12)
    day = st.slider('Day of the month', 1, 31, 15)
    month = st.slider('Month', 1, 12, 6)
    day_of_week = st.slider('Day of the week', 0, 6, 3)
    distance = st.number_input('Distance (km)', min_value=0.1, max_value=100.0, value=5.0)
    passenger_count = st.slider('Number of passengers', 1, 6, 2)
    
    if st.button('Predict Fare'):
        model = load_model()
        input_data = pd.DataFrame({
            'hour': [hour],
            'day': [day],
            'month': [month],
            'day_of_week': [day_of_week],
            'distance': [distance],
            'passenger_count': [passenger_count]
        })
        prediction = model.predict(input_data)
        st.success(f'The predicted fare is ${prediction[0]:.2f}')

if __name__ == '__main__':
    main()