import streamlit as st
import pandas as pd
import joblib

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
    st.header('User Input Features')

    st.write('Please select the values for each feature below. The selected values will be displayed in the table to the right.')

    st.subheader('1. Senior Citizen')
    st.write('Is the customer a senior citizen?')
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="senior_citizen")

    st.subheader('2. Tech Support')
    st.write('Does the customer have technical support?')
    TechSupport = st.selectbox("Tech Support", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="tech_support")

    st.subheader('3. Contract Type')
    st.write('What type of contract does the customer have?')
    Contract = st.selectbox(
        "Contract Type", 
        [0, 1, 2], 
        format_func=lambda x: ["Month-to-month", "One year", "Two year"][x],
        key="contract_type"
    )

    st.subheader('4. Internet Service')
    st.write('What type of internet service does the customer have?')
    InternetService = st.selectbox(
        "Internet Service", 
        [0, 1, 2],
        format_func=lambda x: ["No service", "DSL", "Fiber optic"][x],
        key="internet_service"
    )

    st.subheader('5. Total Charges')
    st.write('What are the total charges for the customer\'s service?')
    TotalCharges = st.text_input("Total Charges", "1000.0", key="total_charges")

    st.subheader('6. Payment Method')
    st.write('What is the customer\'s payment method?')
    PaymentMethod = st.selectbox(
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
    # Make prediction using the model
    try:
        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)[0]  # Get probabilities for both classes

        st.subheader("Prediction Result:")
        st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
        st.write(f"Prediction Probability: Churn: {prediction_proba[1]:.2f}, No Churn: {prediction_proba[0]:.2f}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


st.markdown("© BML Group, 2024")
