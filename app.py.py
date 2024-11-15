import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained Logistic Regression model (updated filename)
try:
    model = joblib.load('logistic_regression_pipeline.pkl')  # Make sure this matches the model file name
    st.sidebar.success("Model loaded successfully!")
    st.sidebar.write(f"Model type: {type(model)}")  # This will show the model's type
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Define the list of selected features
selected_features = ['SeniorCitizen', 'TechSupport', 'Contract', 'InternetService', 'TotalCharges', 'PaymentMethod']

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

# Ensure the input data contains the selected features, in the correct order
input_df = input_df[selected_features]

# Display user input
st.subheader("User Input:")
st.write(input_df)

# Predict using the trained model
try:
    # Check if the model is the correct type
    if hasattr(model, 'predict'):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[0]  # Get probabilities for both classes

        st.subheader("Prediction Result:")
        st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
        st.write(f"Prediction Probability: Churn: {prediction_proba[1]:.2f}, No Churn: {prediction_proba[0]:.2f}")
    else:
        st.error("Loaded model is not a valid LogisticRegression model.")
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
