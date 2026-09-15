import streamlit as st
import numpy as np
import pickle

# Load model
with open("model_rf_param.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler if used
try:
    with open("scalar.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

st.title("Health Insurance Cost Prediction")
st.write("Enter the customer details to predict insurance cost")

# User inputs
age = st.number_input("Age", min_value=0, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
#bloodpressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)

gender = st.selectbox("Gender", ["Female", "Male"])
#diabetic = st.selectbox("Diabetic", ["No", "Yes"])
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Manual encoding (same as get_dummies)
gender_male = 1 if gender == "Male" else 0
#diabetic_yes = 1 if diabetic == "Yes" else 0
smoker_yes = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0
# northeast → all zeros

# Combine inputs
input_data = np.array([[
    age, bmi, children,
    gender_male,  smoker_yes,
    region_northwest, region_southeast, region_southwest
]])

# Apply scaling if used
if scaler:
    input_data = scaler.transform(input_data)

# Predict
if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Insurance Cost: ₹ {prediction[0]:,.2f}")
