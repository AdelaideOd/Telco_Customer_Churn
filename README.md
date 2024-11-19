# Customer Churn Prediction for BML Group  📊

## About BML Group  
BML Group is a fictional telecom company at the forefront of innovation, dedicated to enhancing connectivity and improving customer experiences. With a focus on customer-centric solutions, BML Group strives to deliver exceptional value while fostering long-term relationships with its subscribers.  

## Introduction  
This project aims to predict customer churn for a telecom company using machine learning techniques. Customer churn, where customers stop using a service, is a critical problem in the telecom industry. By identifying at-risk customers, telecom companies can take proactive measures to improve retention and customer satisfaction. 

## About This Project  

This repository is part of a collaborative project created by a group of Data Analysts as a capstone for the Blossom Academy Data Analytics Fellowship. Our goal was to develop predictive models to assess **customer churn** in the telecom industry. This project served as both a practical application of our skills and a showcase of the power of data-driven decision-making.  

Through rigorous analysis and modeling, we aimed to deliver actionable insights that can help telecom companies proactively identify at-risk customers and implement effective retention strategies. The project was also proudly presented during our graduation ceremony, highlighting our dedication and teamwork in addressing real-world challenges.

### Connect with Us on LinkedIn
Scan the QR code below to visit our LinkedIn profiles:

![Our LinkedIn Profiles](Our%20LinkedIn%20Profiles.png)



---

## Project Goals  
- Analyze customer data to understand key factors influencing churn.  
- Develop, train, and evaluate various machine learning models to predict churn.  
- Deploy the best-performing model using **Streamlit** to provide a user-friendly web application for churn prediction.

---

## Dataset  
The dataset used in this project contains customer information such as:  
- **Demographic details**  
- **Service subscription details**  
- **Usage patterns**  
- **Payment methods**  
- **Churn status**  

**Key Features:**  
- **Rows and Columns:** Before preprocessing: 7043 rows and 21 columns| After preprocessing:7032 rows and 20 columns  
- **Target Variable:** `Churn` (Binary - 0: Non-churning, 1: Churning)  

---

## Workflow  

### 1. Data Preprocessing  
- Handled missing values and outliers.  
- Encoded categorical variables using **LabelEncoder** and **One-Hot Encoding**.
- Scaled numerical features using **RobustScaler**.  

### 2. Exploratory Data Analysis  
- Visualized customer distribution across various features.  
- Analyzed correlations and feature importance.  

### 3. Machine Learning Models  
Trained and evaluated multiple models, including:  
- Logistic Regression  
- Random Forest 
- Support Vector Machine (SVC)  
- Voting Classifier  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  

### 4. Model Evaluation  
- Used **F1-score** as the primary evaluation metric due to class imbalance.  
- Compared models using metrics like **ROC-AUC** and confusion matrices.  
- **Best-performing model:** Logistic Regression  

### 5. Deployment  
- Deployed the Logistic Regression model using **Streamlit**, providing an interactive web app for churn prediction.  

## Usage   
https://telcocustomerchurn--bml.streamlit.app/

---

## Technologies Used  
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, , Matplotlib, Seaborn, Streamlit  
- **Deployment:** Streamlit  

---

## Future Work  
- Incorporate more features into the dataset.  
- Experiment with additional algorithms like Neural Networks.
- Employ hyperparameter tuning and smoothing. 
- Enhance the Streamlit app with more interactive visualizations and reports.  

---

## Acknowledgments  
This project was part of the **Generation Ghana and Blossom Academy Data Analytics Fellowship** and sponsored by the **MasterCard Foundation**.  
Special thanks to our instructors for their guidance.  
