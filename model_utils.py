import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model_and_scaler(model_path="best_rf_model.joblib", scaler_path="scaler.joblib"):
    """
    Loads the trained model and scaler.

    Args:
        model_path (str): Path to the saved model file.
        scaler_path (str): Path to the saved scaler file.

    Returns:
        tuple: A tuple containing the loaded model and scaler objects.
               Returns (None, None) if files are not found.
    """
    model = None
    scaler = None
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}!")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")

    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully from {scaler_path}!")
    except FileNotFoundError:
         print(f"Warning: Scaler file '{scaler_path}' not found. You may need to scale your data manually or ensure the scaler is saved.")

    return model, scaler

def predict_status(model, scaler, data):
    """
    Makes predictions on new data using the loaded model and scaler.

    Args:
        model: The trained model object.
        scaler: The fitted scaler object.
        data (pd.DataFrame): DataFrame containing the new data for prediction.

    Returns:
        list: A list of predicted statuses ('Dropout' or 'Graduate').
              Returns None if the model is not loaded.
    """
    if model is None:
        print("Error: Model is not loaded. Cannot make predictions.")
        return None

    # Ensure the data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Scale the data using the provided scaler
    # If scaler is None, this step will be skipped, but it's crucial for accurate predictions
    if scaler is not None:
        data_scaled = scaler.transform(data)
    else:
        print("Warning: Scaler not provided. Predictions will be made on unscaled data.")
        data_scaled = data


    # Make predictions
    predictions = model.predict(data_scaled)

    # Map predictions back to original labels
    status_map = {0: 'Dropout', 1: 'Graduate'}
    predicted_status = [status_map[pred] for pred in predictions]

    return predicted_status

if __name__ == "__main__":
    # This block will only run when model_utils.py is executed directly

    # Example usage:
    print("Running example usage from model_utils.py")

    # Load the model and scaler
    # Make sure your model and scaler files are in the same directory
    model, scaler = load_model_and_scaler()

    if model is not None:
        # Create some sample data for testing
        # Replace this with your actual new data
        sample_data = {
            'Marital_status': [1, 2, 1],
            'Application_mode': [1, 2, 1],
            'Application_order': [1, 2, 1],
            'Course': [1, 5620, 9500],
            'Daytime_evening_attendance': [1, 0, 1],
            'Previous_qualification': [1, 1, 1],
            'Previous_qualification_grade': [25, 50, 13], 
            'Nationality': [1, 1, 1],
            'Mothers_qualification': [1, 12, 22],
            'Fathers_qualification': [2, 13, 44],
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

        # Convert sample data to DataFrame (if it's not already)
        sample_df = pd.DataFrame(sample_data)

        # Make predictions
        predicted_status = predict_status(model, scaler, sample_df)

        # Print the predicted status
        if predicted_status is not None:
            print("\nPredicted Status for the sample data:")
            print(predicted_status)

    else:
        print("Model not loaded. Skipping prediction example.")