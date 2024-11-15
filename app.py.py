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

# Display user input
st.subheader("User Input:")
st.write(input_df)

# Ensure valid numeric inputs by explicitly converting to numeric and handling non-numeric cases
try:
    # Step 1: Convert the columns to numeric explicitly, ensuring that they are in the right format.
    input_df = input_df[selected_features].apply(pd.to_numeric, errors='coerce')  # Coerce non-numeric values to NaN

    # Log the intermediate state of the data
    st.write("Data after conversion to numeric (handling errors):")
    st.write(input_df)

    # Step 2: Handle NaN values by replacing them with zero.
    input_df = input_df.fillna(0)  # Replace NaN values with 0 to avoid errors
    
    # Log the data after NaN handling
    st.write("Data after handling NaN values (replaced with 0):")
    st.write(input_df)
    
    # Check if there are any remaining NaN values or invalid entries
    if input_df.isnull().values.any():
        raise ValueError("There are still missing values in the input data.")

    # Step 3: Ensure the data is in the correct type (numeric)
    st.write("Data types of input features:")
    st.write(input_df.dtypes)
    
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
