# Importing Libraries -->

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ------------------------------

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("knn_model.pkl")

# Features to scale
scaled_features = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']

# Sreamlit code  -->


# Feature names and descriptions for dropdowns
features = {
    'age': 'Age',
    'sex': 'Sex',
    'chest_pain_type': 'Chest Pain Type',
    'resting_blood_pressure': 'Resting Blood Pressure',
    'cholesterol': 'Serum Cholesterol in mg/dl',
    'fasting_blood_sugar': 'Fasting Blood Sugar > 120 mg/dl',
    'rest_ecg': 'Resting Electrocardiographic Results',
    'max_heart_rate_achieved': 'Maximum Heart Rate Achieved',
    'exercise_induced_angina': 'Exercise Induced Angina',
    'st_depression': 'Oldpeak',
    'st_slope': 'The Slope of the Peak Exercise ST Segment',
    'num_major_vessels': 'Number of Major Vessels Colored by Fluoroscopy',
    'thalassemia': 'Thalassemia'
}

# Options for categorical features
options = {
    'sex': {'Male': 1, 'Female': 0},
    'chest_pain_type': {
        'Typical Angina': 0,
        'Atypical Angina': 1,
        'Non-anginal Pain': 2,
        'Asymptomatic': 3
    },
    'fasting_blood_sugar': {
        '> 120 mg/dl': 1,
        '<= 120 mg/dl': 0
    },
    'rest_ecg': {
        'Normal': 0,
        'Having ST-T wave abnormality': 1,
        'Showing probable or definite left ventricular hypertrophy': 2
    },
    'exercise_induced_angina': {
        'Yes': 1,
        'No': 0
    },
    'st_slope': {
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    },
    'num_major_vessels': {
        '0 major vessels': 0,
        '1 major vessel': 1,
        '2 major vessels': 2,
        '3 major vessels': 3
    },
    'thalassemia': {
        'Normal': 0,
        'Fixed defect': 1,
        'Reversible defect': 2
    }
}

# Title

st.title("Heart Disease Prediction ❤️")

# CSS for cursor style
st.markdown(
    """
    <style>
    .stSelectbox > div > div:first-child {
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input fields
input_features = {}

# Create a layout with at least 2 columns per row
cols = st.columns(2)

for i, (feature, label) in enumerate(features.items()):
    col = cols[i % 2]
    if feature in options:
        selected_option = col.selectbox(label, list(options[feature].keys()), key=feature,placeholder="Select.......",index=None)
        if selected_option:
            input_features[feature] = options[feature][selected_option]
    elif feature=='age':
        input_features[feature] =col.number_input(label=label, placeholder="Input....", value=None,max_value=100,min_value=0)
    elif feature=='resting_blood_pressure':
        input_features[feature] =col.number_input(label=label, placeholder="Input....", value=None,min_value=200,max_value=94)
    elif feature=='cholesterol':
        input_features[feature] =col.number_input(label=label, placeholder="Input....", value=None,min_value=126,max_value=564)
    elif feature=='max_heart_rate_achieved':
        input_features[feature] =col.number_input(label=label, placeholder="Input....", value=None,max_value=202,min_value=71)
    elif feature=='st_depression':
        input_features[feature] =col.number_input(label=label, placeholder="Input....", value=None,max_value=7,min_value=0)
    
# ---------------------

# Prediction Code  -->


# Convert input to DataFrame
dic = {key: [value] for key, value in input_features.items()}
df = pd.DataFrame(data=dic)

# Scale specific features
df[scaled_features] = scaler.transform(df[scaled_features])

# Add a button to make prediction
if st.button("Predict"):
    # Predict
    prediction = model.predict(df.iloc[0].values.reshape(1, -1))

    # Display prediction
    st.write("Prediction:", "Positive: You must go and see a doctor immediately" if prediction[0] == 1 else "Negative: You don't have a Heart Disease but you must also consult a doctor. ")
