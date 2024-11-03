
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pre-trained model
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))
car = pd.read_csv("clean_quiker.csv")

# Set page configuration
st.set_page_config(page_title="Quiker Car Predictor", page_icon="🚗")

# Custom CSS for theme and centering
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f4ff;  /* Light blue background */
    }
    .form-container {
        background-color: #e0f7fa;  /* Lighter blue */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.title("Quicker Car Predictor")

# Centering the form in the middle of the page
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    # Sidebar for user inputs
    st.header("Car Details")

    # Get unique values for dropdowns
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    years = sorted(car["year"].unique(), reverse=True)
    fuels = sorted(car["fuel"].unique())

    # User input fields
    selected_company = st.selectbox("Select Company", companies)
    selected_model = st.selectbox("Select Car Model", car_models)
    selected_year = st.selectbox("Select Year", years)
    selected_fuel = st.selectbox("Select Fuel Type", fuels)
    kilometers_driven = st.number_input("Kilometers Driven", min_value=0, value=0)

    # Prediction button
    if st.button("Predict Price"):
        input_data = {
            "company": [selected_company],
            "name": [selected_model],
            "year": [selected_year],
            "fuel": [selected_fuel],
            "km": [kilometers_driven]
        }
        input_df = pd.DataFrame(input_data)

        # Make prediction
        prediction = model.predict(input_df)

        # Display result
        st.success(f"The predicted price for the car is: ₹{np.round(prediction[0], 2)}")

    st.markdown('</div>', unsafe_allow_html=True)
