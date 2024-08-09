import streamlit as st
import pandas as pd
import os
import pickle
from datetime import datetime

html_temp = """
<div style="background-color:yellow;padding:1.5px">
<h1 style="color:black;text-align:center;">House Price Prediction</h1>
</div><br>"""
st.markdown(html_temp, unsafe_allow_html=True)

#getting the model and scaler path
model_path = os.path.abspath(os.path.join(os.getcwd(), "catboost_model.pkl"))
scaler_path = os.path.abspath(os.path.join(os.getcwd(), "Scaler.pkl"))

with open(model_path, 'rb') as model_file:
    cat_model = pickle.load(model_file)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

def main():
    # Title
    st.title("🏠 House Price Predictor")
    
    # Subtitle
    st.write("Enter the details of the house to predict its price.")
    
    # Input fields for user to input data
    lat = st.sidebar.number_input("Enter the latitude coordinate :", min_value=47.1559, max_value=47.7776, value=47.5, help="Latitude Coordinate of House")
    living_measure = st.sidebar.number_input("Enter square footage of the house:", min_value=290.0, max_value=13540.0, value=5000.0, help="Square Footage of House")
    long_converted = st.sidebar.number_input("Enter the longitude coordinate:", min_value=-122.519, max_value=-121.315, value=-122.0, help="Longitude Coordinate of House")
    quality = st.sidebar.number_input("What is the quality of the house:", min_value=1.0, max_value=13.0, value=9.0, help="Grade of House, from Grading System")
    furnished = st.sidebar.selectbox("Is the house furnished?:", ('Yes','No'), help="Presence of Amenities")
    if furnished == 'Yes':
        furnished = 1
    else:
        furnished = 0
        
    lot_measure15 = st.sidebar.number_input("What is the lot size area?:", min_value=651.0, max_value=871200.0, value=200000.0,help="Lot Size Area As At 2015")
    ceil_measure = st.sidebar.number_input("Enter square footage of house excluding basement:", min_value=290.0, max_value=9410.0, value=5000.0, help=" Square Footage ")

    current_year = datetime.now().year
    
    yr_built_converted = st.sidebar.number_input("Enter the year house was built:", min_value=1900.0, max_value=2015.0, value=2010.0, help="Year Of House Construction.")
    
    if yr_built_converted > current_year:
        st.warning(f"Year built cannot be in the future. Please enter a value less than or equal to {current_year}.")
        
    coast_converted = st.sidebar.selectbox("Does the house have a waterfront?:", ('Yes','No'),help="Presence of Waterfront")
    if coast_converted == 'Yes':
        coast_converted = 1
    else:
        coast_converted = 0
    sight = st.sidebar.number_input("Enter value for sight:", min_value=0.0, max_value=4.0, value=2.0, help="Times House Has Been Viewed")
    zipcode = st.sidebar.number_input("Enter the house zipcode:", min_value=98001, max_value=98199, value=98006, help="Zipcode of House Location")

    # Predict button
    if st.button("Predict Price"):
        try:
            # Create a DataFrame with user input
            input_data = pd.DataFrame([[lat, living_measure, long_converted, quality, furnished, lot_measure15, ceil_measure, yr_built_converted, coast_converted, sight, zipcode]],
                                      columns=['lat', 'living_measure', 'long_converted', 'quality', 'furnished', 'lot_measure15', 'ceil_measure', 'yr_built_converted', 'coast_converted', 'sight', 'zipcode'])
            
            # Apply the scaler to the input data
            input_data_scaled = scaler.transform(input_data)
            
            # Predict the price
            prediction = cat_model.predict(input_data_scaled)[0]
            
            # Display the result
            st.success(f"The Estimated Price of The House is: ${prediction:,.2f}")

        except Exception as e:
            st.warning(f"Something went wrong: {e}")

if __name__ == "__main__":
    main()
