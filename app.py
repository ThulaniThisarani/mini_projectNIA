import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import plotly.express as px

st.set_page_config(page_title="Oral Cancer Prediction", layout="centered", page_icon="🦷")

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("oral_cancer_prediction_dataset_modified.csv")
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Impute
    df[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df[numeric_cols])
    df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols])

    # Outlier clipping
    df['Age'] = df['Age'].clip(15, 100)
    df['Cancer Stage'] = df['Cancer Stage'].clip(0, 4)
    df['Survival Rate (5-Year, %)'] = df['Survival Rate (5-Year, %)'].clip(0, 100)

    # Scale numerics
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode categoricals
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders, scaler

# Load and train
df_cleaned, label_encoders, scaler = load_data()
X = df_cleaned.drop(columns=["Oral Cancer (Diagnosis)", "ID"])
y = df_cleaned["Oral Cancer (Diagnosis)"]
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 🦷 Title
st.markdown("<h1 style='text-align: center; color: #0099ff;'>🦷 Oral Cancer Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill out the form below to estimate the probability of oral cancer.</p>", unsafe_allow_html=True)
st.divider()

# 🌟 Sidebar Info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This tool uses machine learning to predict the likelihood of oral cancer based on patient details.")
    st.write("Model: *Random Forest*")
    st.markdown("---")
    st.caption("Developed for IT41033 - Intake11")

# 📝 User Input Form
st.subheader("📋 Enter Patient Information")

def user_input():
    col1, col2 = st.columns(2)
    inputs = {}

    for idx, col in enumerate(X.columns):
        if col in label_encoders:
            options = label_encoders[col].classes_.tolist()
            with (col1 if idx % 2 == 0 else col2):
                value = st.selectbox(f"{col.replace('_',' ')}", options)
                inputs[col] = label_encoders[col].transform([value])[0]
        else:
            with (col1 if idx % 2 == 0 else col2):
                value = st.slider(f"{col}", 0.0, 100.0, step=1.0)
                # scale to [0, 1] range based on dataset
                col_min = df_cleaned[col].min()
                col_max = df_cleaned[col].max()
                inputs[col] = (value - col_min) / (col_max - col_min + 1e-6)

    return pd.DataFrame([inputs])

user_df = user_input()

# 🚨 Predict Button
if st.button("🔍 Predict Oral Cancer Risk"):
    pred = model.predict(user_df)[0]
    prob = model.predict_proba(user_df)[0][1]  # Probability of class 1 (cancer)
    prob_no_cancer = 1 - prob

    st.subheader("📊 Prediction Result:")
    if pred == 1:
        st.error("⚠️ *High Risk of Oral Cancer Detected*")
    else:
        st.success("✅ *Low Risk: No Oral Cancer Detected*")

    st.markdown(f"*Probability of Oral Cancer:* {prob:.2%}")
    st.progress(prob)

    # --- Chart 1: Bar Chart ---
    st.markdown("### 📈 Probability Comparison")
    chart_df = pd.DataFrame({
        "Condition": ["No Oral Cancer", "Oral Cancer"],
        "Probability": [prob_no_cancer, prob]
    })
    fig_bar = px.bar(chart_df, x="Condition", y="Probability", 
                     color="Condition", text="Probability", 
                     range_y=[0,1], color_discrete_map={
                         "No Oral Cancer": "green", 
                         "Oral Cancer": "red"
                     })
    fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Chart 2: Donut Chart ---
    st.markdown("### 🎯 Risk Gauge")
    fig_donut = px.pie(chart_df, values="Probability", names="Condition",
                       hole=0.5, color="Condition", 
                       color_discrete_map={"No Oral Cancer": "green", "Oral Cancer": "red"})
    fig_donut.update_traces(textinfo="label+percent")
    st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")
    st.caption("⚠️ This is a prediction tool and *not a medical diagnosis*. Please consult a professional for medical advice.")
