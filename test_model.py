pip install -r requirements.txt
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
try:
    model = joblib.load("best_rf_model.joblib")
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'best_rf_model.joblib' not found. Make sure the model file is in the same directory.")
    exit()

# Create some sample data for testing
# Replace this with your actual new data in a similar format as your training data
# Make sure the column names and order are the same as your training data
sample_data = {
    'Marital_status': [1, 2, 1],
    'Application_mode': [1, 2, 1],
    'Course': [1, 5620, 9500],
    'Daytime_evening_attendance': [1, 0, 1],
    'Previous_qualification': [1, 1, 1],
    'Nationality': [1, 1, 1],
    'Mother_occupation': [3, 3, 3],
    'Father_occupation': [3, 3, 3],
    'Admission_grade': [140, 130, 150],
    'Displaced': [1, 0, 1],
    'Educational_special_needs': [0, 0, 0],
    'Debtor': [0, 1, 0],
    'Tuition_fees_up_to_date': [1, 0, 1],
    'Gender': [1, 0, 1],
    'Scholarship_holder': [0, 1, 0],
    'Age_at_enrollment': [20, 25, 19],
    'International': [0, 0, 0],
    'Curricular_units_1st_sem_credited': [0, 0, 0],
    'Curricular_units_1st_sem_enrolled': [6, 6, 6],
    'Curricular_units_1st_sem_evaluations': [7, 6, 7],
    'Curricular_units_1st_sem_approved': [6, 4, 6],
    'Curricular_units_1st_sem_grade': [14, 12, 15],
    'Curricular_units_1st_sem_without_evaluations': [0, 0, 0],
    'Curricular_units_2nd_sem_credited': [0, 0, 0],
    'Curricular_units_2nd_sem_enrolled': [6, 6, 6],
    'Curricular_units_2nd_sem_evaluations': [7, 6, 7],
    'Curricular_units_2nd_sem_approved': [6, 4, 6],
    'Curricular_units_2nd_sem_grade': [14, 12, 15],
    'Curricular_units_2nd_sem_without_evaluations': [0, 0, 0],
    'Unemployment_rate': [10.8, 11.2, 10.5],
    'Inflation_rate': [1.4, 1.5, 1.3],
    'GDP': [179.0, 181.0, 178.0]
}

# Convert the sample data to a pandas DataFrame
sample_df = pd.DataFrame(sample_data)

# It's crucial to apply the same preprocessing (scaling) as used during training
# You need to have access to the fitted scaler object or re-fit it on your training data
# For demonstration, we will create a dummy scaler here. In a real scenario, load your trained scaler.
# scaler = joblib.load("scaler.joblib") # Load your trained scaler

# If you don't have the trained scaler saved, you can create a new one and fit it to the sample data
# This is not ideal for production as the scaling should be consistent with training
# A better approach is to save the fitted scaler during training and load it here.
scaler = StandardScaler()
# In a real application, you would fit the scaler on your training data and save it.
# Then load it here and transform the new data.
# For this example, we'll just fit it on the sample data (not recommended for production)
try:
    # This assumes you have a scaler saved.
    scaler = joblib.load("scaler.joblib")
    print("Scaler loaded successfully!")
except FileNotFoundError:
     print("Warning: Scaler not found. Creating a new scaler and fitting on sample data. For production, save and load your trained scaler.")
     # Fit on the sample data - NOT RECOMMENDED FOR PRODUCTION USE
     scaler.fit(sample_df)


sample_data_scaled = scaler.transform(sample_df)

# Make predictions
predictions = model.predict(sample_data_scaled)

# Print the predictions
print("\nPredictions for the sample data:")
print(predictions)

# You can map the predictions back to the original labels if needed
# dropout = 0, graduate = 1
status_map = {0: 'Dropout', 1: 'Graduate'}
predicted_status = [status_map[pred] for pred in predictions]
print("\nPredicted Status:")
print(predicted_status)