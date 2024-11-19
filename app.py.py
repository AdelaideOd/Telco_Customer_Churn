import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

# Function to load Lottie animation from a URL
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation URLs
lottie_loading = "https://lottie.host/0e5e5fd2-62d9-406e-9733-e393d8ae38c1/bYbgzbADHS.json"
lottie_success = "https://lottie.host/748445dc-0823-444f-8cd1-6629ccc7d42d/rEsovbxROq.json"

# Load Lottie animations
loading_animation = load_lottie_url(lottie_loading)
success_animation = load_lottie_url(lottie_success)

# Load the full pipeline with preprocessing and model
try:
    pipeline = joblib.load('logistic_regression_pipeline.pkl')
except Exception as e:
    st.error(f"Error loading the pipeline: {e}")
    st.stop()

# Display the logo
st.image('bml_logo.png', width=200)

# Display the heading
st.title("BML Group")

# Define the selected features for input
selected_features = ['SeniorCitizen', 'TechSupport', 'Contract', 'InternetService', 'TotalCharges', 'PaymentMethod']

# Function to gather user input
def user_input_features():
    st.sidebar.header('User Input Features')

    st.sidebar.write('Please select the values for each feature below. The selected values will be displayed in the table to the right.')

    st.sidebar.subheader('1. Senior Citizen')
    st.sidebar.write('Is the customer a senior citizen?')
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="senior_citizen")

    st.sidebar.subheader('2. Tech Support')
    st.sidebar.write('Does the customer have technical support?')
    TechSupport = st.sidebar.selectbox("Tech Support", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="tech_support")

    st.sidebar.subheader('3. Contract Type')
    st.sidebar.write('What type of contract does the customer have?')
    Contract = st.sidebar.selectbox(
        "Contract Type", 
        [0, 1, 2], 
        format_func=lambda x: ["Month-to-month", "One year", "Two year"][x],
        key="contract_type"
    )

    st.sidebar.subheader('4. Internet Service')
    st.sidebar.write('What type of internet service does the customer have?')
    InternetService = st.sidebar.selectbox(
        "Internet Service", 
        [0, 1, 2],
        format_func=lambda x: ["No service", "DSL", "Fiber optic"][x],
        key="internet_service"
    )

    st.sidebar.subheader('5. Total Charges')
    st.sidebar.write('What are the total charges for the customer\'s service?')
    TotalCharges = st.sidebar.text_input("Total Charges", "1000.0", key="total_charges")

    st.sidebar.subheader('6. Payment Method')
    st.sidebar.write('What is the customer\'s payment method?')
    PaymentMethod = st.sidebar.selectbox(
        "Payment Method", 
        [0, 1, 2, 3],
        format_func=lambda x: [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ][x],
        key="payment_method"
    )
    
    data = {
        "SeniorCitizen": [SeniorCitizen],
        "TechSupport": [TechSupport],
        "Contract": [Contract],
        "InternetService": [InternetService],
        "TotalCharges": [float(TotalCharges)],  # Convert to float
        "PaymentMethod": [PaymentMethod],
    }

    return pd.DataFrame(data)

# Gather user input
input_df = user_input_features()

# Display user input
st.subheader("Customer Profile")
st.write(input_df)

# Create a button to run the prediction
if st.button('Run Prediction'):
    # Display loading animation while prediction is being made
    st_lottie(loading_animation, height=200, key="loading")

    # Make prediction using the model
    try:
        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)[0]  # Get probabilities for both classes

        # Display prediction result
        st.subheader("Prediction Result:")
        st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
        st.write(f"Prediction Probability: Churn: {prediction_proba[1]:.2f}, No Churn: {prediction_proba[0]:.2f}")
        
        # Show success animation after prediction is displayed
        st_lottie(success_animation, height=200, key="success")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Footer
st.markdown("© BML Group, 2024")
