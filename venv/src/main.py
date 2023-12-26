import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split #model training

# Absolute path to modules
src_dir = 'D:/PhishingDetectionTool/venv/src'
sys.path.append(src_dir)


from email_reader import read_email
from data_cleaning import clean_email_body 
from feature_extraction import extract_features


def process_emails(directory, label):
    data = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.eml'):
                file_path = os.path.join(dirpath, filename)
                email_msg = read_email(file_path)

                # Processes the email depending on whether it is multipart
                if email_msg.is_multipart():
                    for part in email_msg.walk():
                        if part.get_content_type() in ['text/plain', 'text/html']:
                            payload = part.get_payload(decode=True)
                            if payload:
                                clean_content = clean_email_body(payload)
                                features = extract_features(file_path, clean_content)
                                features['label'] = label
                                data.append(features)
                                break  # Exits after processing the first relevant part
                else:
                    payload = email_msg.get_payload(decode=True)
                    if payload:
                        clean_content = clean_email_body(payload)
                        features = extract_features(file_path, clean_content)
                        features['label'] = label
                        data.append(features)

    return data



# Directories containing .eml files
phishing_directory = r'D:\PhishingDetectionTool\venv\datasets\PhishingEmails'
legitimate_directory = r'D:\PhishingDetectionTool\venv\datasets\LegitEmailseml'

# Processes emails
phishing_data = process_emails(phishing_directory, 'phishing')
legitimate_data = process_emails(legitimate_directory, 'legitimate')

# Combines and saves to a DataFrame
all_data = phishing_data + legitimate_data
df = pd.DataFrame(all_data)

# Output directory for all datasets
output_dir = r'D:\PhishingDetectionTool\venv\datasets'

# Save the combined dataset in the datasets folder
combined_csv_path = os.path.join(output_dir, 'combined_email_features.csv')
df.to_csv(combined_csv_path, index=False)

# Separate features and target variable
X = df.drop('label', axis=1)
y = df['label']

# Splits the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Saves the sets
train_df = pd.concat([X_train, y_train], axis=1)
train_df.to_csv(os.path.join(output_dir, 'training_set.csv'), index=False)


val_df = pd.concat([X_val, y_val], axis=1)
val_df.to_csv(os.path.join(output_dir, 'validation_set.csv'), index=False)


test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv(os.path.join(output_dir, 'testing_set.csv'), index=False)
