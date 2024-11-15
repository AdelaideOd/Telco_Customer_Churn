import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained pipeline (model + preprocessor)
try:
    pipeline = joblib.load('logistic_regression_model1.pkl')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Display the logo
st.image('bml_logo.png', width=200)

# Display the heading
st.title("BML Group")

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
    TotalCharges = st.sidebar.text_input("Total Charges", "1000.0")
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
    # Use the pipeline's preprocessing to transform the input
    input_df_transformed = pipeline.named_steps['preprocessor'].transform(input_df)

    # Make prediction using the pipeline
    prediction = pipeline.predict(input_df_transformed)
    prediction_proba = pipeline.predict_proba(input_df_transformed)[0]  # Get probabilities for both classes

    st.subheader("Prediction Result:")
    st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
    st.write(f"Prediction Probability: Churn: {prediction_proba[1]:.2f}, No Churn: {prediction_proba[0]:.2f}")

except ValueError as ve:
    st.error(f"Error during prediction: {ve}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
