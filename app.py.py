import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score

# Load the pre-trained Logistic Regression model
model = pickle.load(open('logistic_model.pkl', 'rb'))

# Sidebar for user input features
st.sidebar.header("User Input Features")

def user_input_features():
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    TechSupport = st.sidebar.selectbox("Tech Support", [0, 1], format_func=lambda x: "Yes" if x == 0 else "No")
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
    TotalCharges = st.sidebar.slider("Total Charges", 0.0, 5000.0, 1000.0)  # Adjust range as needed
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
    
    # Input data in a dictionary
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

# Make predictions
try:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display prediction
    st.subheader("Prediction:")
    st.write("Churn" if prediction[0] == 1 else "No Churn")

    # Display prediction probabilities
    st.subheader("Prediction Probability:")
    st.write(f"Churn: {prediction_proba[0][1]:.2f}, No Churn: {prediction_proba[0][0]:.2f}")

except Exception as e:
    st.error(f"An error occurred during prediction: {e}")

# Check Model Accuracy (Optional if test data is available)
st.sidebar.subheader("Evaluate Model (Optional)")

if st.sidebar.checkbox("Display Model Performance"):
    try:
        # If test data is available, uncomment the below lines to evaluate model
        # test_data = pd.read_csv("test_data.csv")
        # X_test = test_data[selected_features]
        # y_test = test_data['Churn']

        # Example: y_pred = model.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)

        # Mock values (replace with calculated accuracy)
        accuracy = 0.85  # Replace with calculated accuracy
        f1 = 0.80       # Replace with calculated F1 score

        # Display model performance
        st.subheader("Model Performance:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
    except Exception as e:
        st.error(f"Could not calculate performance metrics: {e}")
