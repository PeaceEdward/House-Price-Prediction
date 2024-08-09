import streamlit as st
import joblib
import pandas as pd
import os
import pickle

model_path = os.path.abspath(os.path.join(os.getcwd(), "catboost_model.pkl"))
scaler_path = os.path.abspath(os.path.join(os.getcwd(), "Scaler.pkl"))

with open(model_path, 'rb') as model_file:
    cat_model = pickle.load(model_file)

# Load the model using CatBoost's method
# cat_model = CatBoostRegressor()
# cat_model.load_model(model_path)


with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)


# Title
st.title("House Price Predictor")

# Subtitle
st.write("Enter the details of the house to predict its price.")

# Input fields for user to input data
lat = st.number_input("Enter value for latitude (min: 47.1559, max: 47.7776):", min_value=47.1559, max_value=47.7776, value=47.5)
living_measure = st.number_input("Enter value for living_measure (min: 290.0, max: 13540.0):", min_value=290.0, max_value=13540.0, value=5000)
long_converted = st.number_input("Enter value for long_converted (min: -122.519, max: -121.315):", min_value=-122.519, max_value=-121.315, value=-122)
quality = st.number_input("Enter value for quality (min: 1.0, max: 13.0):", min_value=1.0, max_value=13.0, value=9)
furnished = st.number_input("Enter value for furnished (min: 0.0, max: 1.0):", min_value=0.0, max_value=1.0, value=0)
lot_measure15 = st.number_input("Enter value for lot_measure15 (min: 651.0, max: 871200.0):", min_value=651.0, max_value=871200.0, value=200000)
ceil_measure = st.number_input("Enter value for ceil_measure (min: 290.0, max: 9410.0):", min_value=290.0, max_value=9410.0, value=5000)
yr_built_converted = st.number_input("Enter value for yr_built_converted (min: 1900.0, max: 2015.0):", min_value=1900.0, max_value=2015.0, value=2010)
coast_converted = st.number_input("Enter value for coast_converted (min: 0.0, max: 1.0):", min_value=0.0, max_value=1.0, value=0)
sight = st.number_input("Enter value for sight (min: 0.0, max: 4.0):", min_value=0.0, max_value=4.0, value=2)
zipcode = st.number_input("Enter value for zipcode (min: 98001, max: 98199):", min_value=98001, max_value=98199, value=98006)

# Predict button
if st.button("Predict"):
    # Create a DataFrame with user input
    input_data = pd.DataFrame([[lat, living_measure, long_converted, quality, furnished, lot_measure15, ceil_measure, yr_built_converted, coast_converted, sight, zipcode]],
                              columns=['lat', 'living_measure', 'long_converted', 'quality', 'furnished', 'lot_measure15', 'ceil_measure', 'yr_built_converted', 'coast_converted', 'sight', 'zipcode'])
    
    # Apply the scaler to the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the price
    prediction = cat_model.predict(input_data_scaled)[0]
    
    # Display the result
    st.success(f"Estimated Price: ${prediction:,.2f}")
