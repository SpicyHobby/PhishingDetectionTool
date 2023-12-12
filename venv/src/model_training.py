import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib #model is saved, this allows to load the model later to make predictions without needing to retrain it.
import random

# Loading the dataset
df = pd.read_csv('D:\PhishingDetectionTool\venv\datasets/combined_email_features.csv')

# Preprocessing the data
X = df.drop('label', axis=1)
y = df['label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model using probabilities
probabilities = model.predict_proba(X_test)

# Extract probabilities of the positive class (phishing)
phishing_probabilities = probabilities[:, 1]

# Convert probabilities to percentages
phishing_percentages = phishing_probabilities * 100

# Select a random email and print its phishing likelihood
random_email_index = random.randint(0, len(phishing_percentages) - 1)
print(f"Random Email (Index {random_email_index}) Phishing Likelihood: {phishing_percentages[random_email_index]:.2f}%")

# To display a classification report
print(classification_report(y_test, model.predict(X_test)))

# Save the model with the learnt progress
joblib.dump(model, 'phishing_detection_model.pkl')
