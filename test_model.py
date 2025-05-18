import streamlit as st
import pandas as pd
import numpy as np
# Import the functions from your model_utils.py file
from model_utils import load_model_and_scaler, predict_status

# Define the paths to your model and scaler files
MODEL_PATH = "best_rf_model.joblib"
SCALER_PATH = "scaler.joblib" # Make sure you save your scaler during training!

# Load the model and scaler when the app starts
# Use st.cache_resource to avoid reloading on every rerun of the app
@st.cache_resource
def get_model_and_scaler():
    """Loads the model and scaler using the utility function."""
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    if model is None or scaler is None:
        st.error("Failed to load model or scaler. Please check the file paths and ensure they are saved correctly.")
        return None, None
    return model, scaler

model, scaler = get_model_and_scaler()

# Define the order of features as expected by the model
# Make sure this list matches the order of features in your training data
feature_order = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification',
    'Previous_qualification_grade', 'Nationality', 'Mothers_qualification',
    'Fathers_qualification', 'Mother_occupation', 'Father_occupation',
    'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
    'Age_at_enrollment', 'International',
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate',
    'Inflation_rate', 'GDP'
]


# --- Streamlit App Interface ---
st.title("Student Status Prediction")

st.write("Enter the student's information below to predict their status (Dropout or Graduate).")

# Create input fields for each feature
input_data = {}
for feature in feature_order:
    # You might want to customize the input type based on the feature's data type
    # For simplicity, using number_input for most numeric fields
    # For categorical features, you might use st.selectbox or st.radio
    # You might also need to add help text or descriptions for each feature
    if feature in ['Marital_status', 'Application_mode', 'Daytime_evening_attendance',
                   'Previous_qualification', 'Nationality', 'Mothers_qualification',
                   'Fathers_qualification', 'Mother_occupation', 'Father_occupation',
                   'Displaced', 'Educational_special_needs', 'Debtor',
                   'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
                   'International']:
        # Example: Using number_input for simplicity, but ideally use selectbox for true categories
        input_data[feature] = st.number_input(f"Enter value for '{feature}':", value=0, step=1)
    elif feature in ['Previous_qualification_grade', 'Admission_grade',
                     'Age_at_enrollment', 'Curricular_units_1st_sem_credited',
                     'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
                     'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
                     'Curricular_units_1st_sem_without_evaluations',
                     'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
                     'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
                     'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations']:
        input_data[feature] = st.number_input(f"Enter value for '{feature}':", value=0.0, step=0.1)
    elif feature in ['Unemployment_rate', 'Inflation_rate', 'GDP']:
         input_data[feature] = st.number_input(f"Enter value for '{feature}':", value=0.0, step=0.01)
    else: # Default to text_input or number_input based on your data
         input_data[feature] = st.number_input(f"Enter value for '{feature}':", value=0.0, step=0.1)


# Convert input data to a pandas DataFrame in the correct order
input_df = pd.DataFrame([input_data], columns=feature_order)


# Make prediction when the button is clicked
if st.button("Predict Status"):
    if model is not None and scaler is not None:
        # Make prediction using the predict_status function
        prediction = predict_status(model, scaler, input_df)

        # Display the prediction
        if prediction:
            st.success(f"Predicted Status: **{prediction[0]}**")
        else:
            st.warning("Could not get prediction.")
    else:
        st.error("Model or scaler is not loaded. Please check the console for errors.")
