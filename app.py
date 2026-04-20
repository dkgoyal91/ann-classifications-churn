import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

##load the trained model and the preprocessing objects
model = tf.keras.models.load_model('churn_model.h5')
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder   = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder  = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler          = pickle.load(f)    

## streamlit app


st.title("Customer Churn Prediction")

## create input fields for the user to enter customer data
gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
age = st.slider("Age", 18,92)
tenure = st.slider("Tenure", 0,10)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
## one hot encode the 'Geography' column
geography_encoded = onehot_encoder.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

## create a dataframe with columns in the correct order (as they were during training)
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'EstimatedSalary': [estimated_salary]
})

## concatenate the one-hot encoded geography columns
for col in geography_encoded_df.columns:
    input_df[col] = geography_encoded_df[col].values

# Ensure columns are in the correct order (same as scaler fit)
expected_columns = scaler.get_feature_names_out()
input_df = input_df[expected_columns]

## scaling the input data
input_scaled = scaler.transform(input_df)

## predict the probability of churn
churn_probability = model.predict(input_scaled)

st.write(f"Churn Probability: {churn_probability[0][0]:.2f}")

if churn_probability[0][0] > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")



## streamlit run 03_app.py


