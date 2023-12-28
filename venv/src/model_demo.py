import joblib
import pandas as pd
import os
import random
from email_reader import read_email
from feature_extraction import extract_features
from data_cleaning import clean_email_body
import sys

# Updates with path to scripts
sys.path.append('D:\\PhishingDetectionTool\\venv\\src')

# Loads the trained model, scaler, and feature names
def load_model_scaler_and_features(model_path, scaler_path, feature_names_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    return model, scaler, feature_names

# Processes a single .eml file
def process_email(email_msg, scaler, feature_columns):
    # Clean the email body
    if email_msg.is_multipart():
        cleaned_content = ''
        for part in email_msg.walk():
            if part.get_content_type() in ['text/plain', 'text/html']:
                payload = part.get_payload(decode=True)
                if payload:
                    cleaned_content = clean_email_body(payload)
                    break
    else:
        payload = email_msg.get_payload(decode=True)
        cleaned_content = clean_email_body(payload) if payload else ''

    # Extracts features
    features = extract_features(email_msg, cleaned_content)

    # Initialize a DataFrame with the correct structure
    features_df = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        features_df.loc[0, col] = features.get(col, 0)  # Use feature value or 0 if not found

    # Convert boolean columns to integers, if any
    bool_cols = ['is_secure', 'has_attachment', 'blank_subject',
                 'contains_suspicious_keyword', 'contains_suspicious_keyword_body', 'urgent_tone']
    features_df[bool_cols] = features_df[bool_cols].astype(int)

    # Scales the features
    scaled_features = scaler.transform(features_df)

    return scaled_features

# Main function to run the script
def main():
    # Paths to model, scaler, and feature names
    model_path = 'D:\\PhishingDetectionTool\\venv\\src\\phishing_detection_best_model.pkl'
    scaler_path = 'D:\\PhishingDetectionTool\\venv\\src\\scaler.pkl'
    feature_names_path = 'D:\\PhishingDetectionTool\\venv\\src\\feature_names.pkl'

    # Loads the model, scaler, and feature names
    model, scaler, feature_columns = load_model_scaler_and_features(model_path, scaler_path, feature_names_path)

    # Path to email datasets
    phishing_directory = 'D:\\PhishingDetectionTool\\venv\\datasets\\unused_emails\\phishing'
    legitimate_directory = 'D:\\PhishingDetectionTool\\venv\\datasets\\unused_emails\\legitimate'

    # Chooses a random email from one of the directories
    chosen_directory = random.choice([phishing_directory, legitimate_directory])
    chosen_email = random.choice(os.listdir(chosen_directory))
    email_path = os.path.join(chosen_directory, chosen_email)

    # Processes the email
    email_msg = read_email(email_path)
    processed_email = process_email(email_msg, scaler, feature_columns)

    # Makes a prediction
    prediction = model.predict(processed_email)
    confidence = model.predict_proba(processed_email)

    # Displays the results
    print(f"Email: {chosen_email}")
    print(f"Email Prediction: {'Phishing' if prediction[0] == 1 else 'Legitimate'}")
    print(f"Confidence Score: {confidence[0][prediction[0]]}")

if __name__ == "__main__":
    main()