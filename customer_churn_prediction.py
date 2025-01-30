import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE  # For handling imbalanced data
import xgboost as xgb  # Alternative model
import joblib  # For saving the model

# Load the dataset
df = pd.read_csv('telco_churn.csv')

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Drop unnecessary columns
df.drop(columns=['customerID'], errors='ignore', inplace=True)

# Convert categorical variables to numerical
label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Define features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Train Random Forest model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the best model
joblib.dump(rf_classifier, 'churn_model.pkl')

# Streamlit Dashboard
st.title("Customer Churn Prediction Dashboard")

# Load trained model
model = joblib.load('churn_model.pkl')

def predict_churn(data):
    data = scaler.transform([data])
    prediction = model.predict(data)
    return "Churn" if prediction[0] == 1 else "Not Churn"

# User input for prediction
st.sidebar.header("User Input Features")
user_input = []
for col in X.columns:
    value = st.sidebar.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    user_input.append(value)

if st.sidebar.button("Predict"):
    prediction = predict_churn(user_input)
    st.sidebar.write("Prediction:", prediction)

# Visualization
st.subheader("Churn Distribution")
sns.countplot(x='Churn', data=df)
st.pyplot()

# Confusion Matrix
st.subheader("Confusion Matrix")
y_pred = rf_classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
st.pyplot()

# ROC Curve
st.subheader("ROC Curve")
roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
st.pyplot()
