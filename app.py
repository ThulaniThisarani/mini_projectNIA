import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==================================================
# Streamlit Page Setup
# ==================================================
st.set_page_config(page_title="Oral Cancer Prediction", layout="centered", page_icon="🦷")

st.markdown("<h1 style='text-align: center; color: #0099ff;'>🦷 Oral Cancer Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill out the form below to estimate the probability of oral cancer.</p>", unsafe_allow_html=True)
st.divider()

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This tool uses machine learning to predict the likelihood of oral cancer based on patient details.")
    st.write("Models available: *Random Forest, Logistic Regression, SVM, KNN, Gradient Boosting*")
    st.markdown("---")
    st.caption("Developed for IT41033 - Intake11")

# ==================================================
# Load Data
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv("oral_cancer_prediction_dataset_modified.csv")
    df = df.drop_duplicates()

    # Drop rows where target is missing
    df = df.dropna(subset=["Oral Cancer (Diagnosis)"]).reset_index(drop=True)
    return df

df = load_data()

# ==================================================
# Preprocessing
# ==================================================
X = df.drop(columns=["Oral Cancer (Diagnosis)", "ID"])
y = df["Oral Cancer (Diagnosis)"].map({"No": 0, "Yes": 1})  # Encode target

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features),
    ]
)

# ==================================================
# Train Models
# ==================================================
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

trained_models = {}
for name, clf in models.items():
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])
    pipeline.fit(X, y)
    trained_models[name] = pipeline

# ==================================================
# User Input Form
# ==================================================
st.subheader("📋 Enter Patient Information")

def user_input():
    col1, col2 = st.columns(2)
    inputs = {}

    for idx, col in enumerate(X.columns):
        if col in cat_features:
            options = df[col].dropna().unique().tolist()
            with (col1 if idx % 2 == 0 else col2):
                value = st.selectbox(f"{col}", options)
                inputs[col] = value
        else:
            with (col1 if idx % 2 == 0 else col2):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                value = st.slider(f"{col}", min_val, max_val, float(df[col].median()))
                inputs[col] = value

    return pd.DataFrame([inputs])

user_df = user_input()

# ==================================================
# Prediction
# ==================================================
model_choice = st.radio("Select Model:", list(trained_models.keys()))

if st.button("🔍 Predict Oral Cancer Risk"):
    model = trained_models[model_choice]
    pred = model.predict(user_df)[0]
    prob = model.predict_proba(user_df)[0][1]

    st.subheader("📊 Prediction Result:")
    if pred == 1:
        st.error("⚠️ *High Risk of Oral Cancer Detected*")
    else:
        st.success("✅ *Low Risk: No Oral Cancer Detected*")

    st.markdown(f"*Probability of Oral Cancer:* {prob:.2%}")
    st.progress(prob)

    st.markdown("---")
    st.caption("⚠️ This is a prediction tool and *not a medical diagnosis*. Please consult a professional for medical advice.")
