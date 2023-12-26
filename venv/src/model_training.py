import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import os
import seaborn as sns



# Function to preprocess DataFrame
def preprocess_df(df):
    # Converts boolean columns to numeric
    bool_cols = ['is_secure', 'has_attachment', 'blank_subject', 'contains_suspicious_keyword',
                 'contains_suspicious_keyword_body', 'urgent_tone']
    df[bool_cols] = df[bool_cols].astype(int)

    # Drops non-numeric/string columns
    df = df.drop(columns=['sender_domain', 'urgent_subject'])

    # Encodes the label column
    df['label'] = df['label'].map({'legitimate': 0, 'phishing': 1})

    return df

# Loads the datasets
train_df = pd.read_csv('D:/PhishingDetectionTool/venv/datasets/training_set.csv')
validation_df = pd.read_csv('D:/PhishingDetectionTool/venv/datasets/validation_set.csv')
test_df = pd.read_csv('D:/PhishingDetectionTool/venv/datasets/testing_set.csv')

# Preprocesses the datasets
train_df = preprocess_df(train_df)
validation_df = preprocess_df(validation_df)
test_df = preprocess_df(test_df)

# Saves feature names before scaling
feature_names = train_df.drop('label', axis=1).columns.tolist()
joblib.dump(feature_names, 'D:/PhishingDetectionTool/venv/src/feature_names.pkl')

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df.drop('label', axis=1))
y_train = train_df['label']
X_validation = scaler.transform(validation_df.drop('label', axis=1))
y_validation = validation_df['label']
X_test = scaler.transform(test_df.drop('label', axis=1))
y_test = test_df['label']

# Initialize and hyperparameter tuning the model using validation set
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_validation, y_validation)
best_model = grid.best_estimator_

# Cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")

# Training the best model found with training set
best_model.fit(X_train, y_train)

# Predictions and evaluation with testing set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba)}")

# Precision-Recall Curve data and AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.2f}")

# Define a directory where the plots will be saved
plot_directory = r"D:\PhishingDetectionTool\venv\generatedplots"
os.makedirs(plot_directory, exist_ok=True)  # This will create the directory if it does not exist

# Generate and save the ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(plot_directory, 'roc_curve.png'))
plt.close()

# Plot and save the Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Logistic Regression (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig(os.path.join(plot_directory, 'precision_recall_curve.png'))
plt.close()

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot and save the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(plot_directory, 'confusion_matrix.png'))
plt.close()

# Extracting and printing confusion matrix values
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix Values:")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")

# Show all plots
plt.tight_layout()
plt.show()


# Saves the best model and the scaler
joblib.dump(best_model, 'D:/PhishingDetectionTool/venv/src/phishing_detection_updated_model.pkl')
joblib.dump(scaler, 'D:/PhishingDetectionTool/venv/src/scalerupdated.pkl')
