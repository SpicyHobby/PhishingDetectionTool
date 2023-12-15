import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Loading the datasets
train_df = pd.read_csv('D:/PhishingDetectionTool/venv/datasets/training_set.csv')
validation_df = pd.read_csv('D:/PhishingDetectionTool/venv/datasets/validation_set.csv')
test_df = pd.read_csv('D:/PhishingDetectionTool/venv/datasets/testing_set.csv')

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

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Save the best model
joblib.dump(best_model, 'phishing_detection_best_model.pkl')
