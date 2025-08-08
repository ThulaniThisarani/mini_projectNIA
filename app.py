import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib

# Load and preprocess the dataset (simulate preprocessing pipeline)
@st.cache_data
def load_data():
    df = pd.read_csv("oral_cancer_prediction_dataset_modified.csv")
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Impute missing values
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Handle outliers
    df['Age'] = df['Age'].clip(15, 100)
    df['Cancer Stage'] = df['Cancer Stage'].clip(0, 4)
    df['Survival Rate (5-Year, %)'] = df['Survival Rate (5-Year, %)'].clip(0, 100)

    # Scale numeric data
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode categoricals
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders, scaler

# Load data and prepare model
df_cleaned, label_encoders, scaler = load_data()
X = df_cleaned.drop(columns=["Oral Cancer (Diagnosis)", "ID"])
y = df_cleaned["Oral Cancer (Diagnosis)"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Web app UI
st.title("🦷 Oral Cancer Prediction App")
st.markdown("Enter the patient details below to predict the risk of oral cancer.")

# Collect user input
def user_input():
    inputs = {}

    # Use all features except ID and target
    for col in X.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_.tolist()
            value = st.selectbox(f"{col}:", options)
            inputs[col] = label_encoders[col].transform([value])[0]
        else:
            value = st.number_input(f"{col}:", min_value=0.0, step=0.1)
            # Scale numeric input
            col_min = df_cleaned[col].min()
            col_max = df_cleaned[col].max()
            inputs[col] = (value - col_min) / (col_max - col_min + 1e-5)

    return pd.DataFrame([inputs])

# Predict
user_df = user_input()

if st.button("Predict"):
    prediction = model.predict(user_df)[0]
    proba = model.predict_proba(user_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ Prediction: Oral Cancer Detected (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Prediction: No Oral Cancer (Probability: {proba:.2f})")

