import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE  # For handling imbalanced data
import joblib  # For saving the model

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        return df
    except FileNotFoundError:
        st.error("The dataset file was not found. Please check the file path.")
        return None

df = load_data()

# Handle missing values
st.write("Missing values before handling:")
st.write(df.isnull().sum())

# Fill numeric columns with median and categorical columns with mode
df.fillna(df.select_dtypes(include=np.number).median(), inplace=True)  # Fill numeric columns with median
df.fillna(df.select_dtypes(include='object').mode().iloc[0], inplace=True)  # Fill categorical columns with mode

st.write("Missing values after handling:")
st.write(df.isnull().sum())

# Drop unnecessary columns
df.drop(columns=['customerID'], errors='ignore', inplace=True)

# Convert categorical variables to numerical
label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns

# Use LabelEncoder for binary categorical variables and OneHotEncoder for others
for col in categorical_columns:
    if df[col].nunique() == 2:  # Binary categorical variables
        df[col] = label_encoder.fit_transform(df[col])
    else:  # Nominal categorical variables
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Define features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handle imbalanced data using SMOTE (only on the training set)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Train Random Forest model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

# Save the model
joblib.dump(rf_classifier, 'churn_model.pkl')

# Streamlit Dashboard
st.title("Customer Churn Prediction Dashboard")

# Load trained model
try:
    model = joblib.load('churn_model.pkl')
except FileNotFoundError:
    st.error("The model file was not found. Please ensure the model is saved.")
    st.stop()

# Function to predict churn
def predict_churn(data):
    data = scaler.transform([data])
    prediction = model.predict(data)
    return "Churn" if prediction[0] == 1 else "Not Churn"

# User input for prediction
st.sidebar.header("User Input Features")
user_input = []
for col in X.columns:
    if col in categorical_columns and df[col].nunique() == 2:  # Binary categorical features
        value = st.sidebar.selectbox(f"{col}", options=df[col].unique())
    elif col in categorical_columns:  # Nominal categorical features
        value = st.sidebar.selectbox(f"{col}", options=df[col].unique())
    else:  # Numeric features
        value = st.sidebar.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    user_input.append(value)

if st.sidebar.button("Predict"):
    prediction = predict_churn(user_input)
    st.sidebar.write("Prediction:", prediction)

# Visualization
st.subheader("Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Churn', data=df, ax=ax)
st.pyplot(fig)

# Confusion Matrix
st.subheader("Confusion Matrix")
y_pred = rf_classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve")
roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.table(pd.DataFrame(report).transpose())
