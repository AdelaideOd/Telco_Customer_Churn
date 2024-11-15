import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model pipeline (model + preprocessing steps)
try:
    model = joblib.load('logistic_regression_pipeline.pkl')  # Load the full pipeline
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Define the list of selected features (must match the model's training features)
selected_features = ['SeniorCitizen', 'TechSupport', 'Contract', 'InternetService', 'TotalCharges', 'PaymentMethod']

def user_input_features():
    """Function to collect input from the user through the sidebar"""
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    TechSupport = st.sidebar.selectbox("Tech Support", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    Contract = st.sidebar.selectbox(
        "Contract Type", 
        [0, 1, 2], 
        format_func=lambda x: ["Month-to-month", "One year", "Two year"][x]
    )
    InternetService = st.sidebar.selectbox(
        "Internet Service",
        [0, 1, 2],
        format_func=lambda x: ["No service", "DSL", "Fiber optic"][x]
    )
    TotalCharges = st.sidebar.slider("Total Charges", 0.0, 10000.0, 1000.0)
    PaymentMethod = st.sidebar.selectbox(
        "Payment Method",
        [0, 1, 2, 3],
        format_func=lambda x: [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ][x]
    )
    
    # Collect user input into a DataFrame
    data = {
        "SeniorCitizen": SeniorCitizen,
        "TechSupport": TechSupport,
        "Contract": Contract,
        "InternetService": InternetService,
        "TotalCharges": TotalCharges,
        "PaymentMethod": PaymentMethod,
    }

    return pd.DataFrame(data, index=[0])

# Get the user's input
input_df = user_input_features()

# Ensure the input data contains the selected features in the correct order
input_df = input_df[selected_features]

# Handle NaN or missing values in the input
input_df = input_df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric and handle non-numeric gracefully
input_df = input_df.fillna(0)  # Replace any NaN values with 0 or other suitable values

# Display user input for review
st.subheader("User Input:")
st.write(input_df)

# Prediction
try:
    # Use the pre-trained model to make predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[0]  # Get probabilities for both classes

    st.subheader("Prediction Result:")
    st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
    st.write(f"Prediction Probability: Churn: {prediction_proba[1]:.2f}, No Churn: {prediction_proba[0]:.2f}")
except Exception as e:
    st.error("An error occurred during prediction. Please check your input values.")
    # Log the error for debugging (you can remove this line for production)
    st.error(f"Error details: {e}")
