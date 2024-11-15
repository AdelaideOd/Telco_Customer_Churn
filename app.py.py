import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained Logistic Regression model
try:
    model = joblib.load('logistic_regression_pipeline.pkl')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define the selected features for input
selected_features = ['SeniorCitizen', 'TechSupport', 'Contract', 'InternetService', 'TotalCharges', 'PaymentMethod']

# Function to gather user input
def user_input_features():
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
    
    data = {
        "SeniorCitizen": SeniorCitizen,
        "TechSupport": TechSupport,
        "Contract": Contract,
        "InternetService": InternetService,
        "TotalCharges": TotalCharges,
        "PaymentMethod": PaymentMethod,
    }

    return pd.DataFrame(data, index=[0])

# Gather user input
input_df = user_input_features()

# Ensure valid numeric inputs by explicitly converting to numeric and handling non-numeric cases
try:
    # Step 1: Convert the columns to numeric explicitly, ensuring that they are in the right format.
    # Since we are using selectboxes and sliders, all values should already be numeric, but we can still check.
    for feature in selected_features:
        if input_df[feature].dtype not in [int, float]:
            raise ValueError(f"Input for feature '{feature}' is not numeric.")

    # No need to fill NaN values with zeros since all inputs should be numeric

    # Step 4: Make prediction using the model
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[0]  # Get probabilities for both classes

    st.subheader("Prediction Result:")
    st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
    st.write(f"Prediction Probability: Churn: {prediction_proba[1]:.2f}, No Churn: {prediction_proba[0]:.2f}")

except ValueError as ve:
    st.error(f"Error during prediction: {ve}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

