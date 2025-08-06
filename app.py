import streamlit as st
import pandas as pd
import pickle
import os
import tarfile
import gdown

# STEP 1: Download the model from Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1k20qTxGm1ad_gZb5ev_NlKv4ZMQ6Ko2J"
MODEL_TAR = "model.tar.gz"
MODEL_PATH = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\lung_cancer_survival\\models\\lung_cancer_model.pkl"

@st.cache_resource
def download_and_extract_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_TAR, quiet=False)
        with tarfile.open(MODEL_TAR, "r:gz") as tar:
            tar.extractall()
        st.success("Model downloaded and extracted.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = download_and_extract_model()

# UI
st.title("🫁 Lung Cancer Survival Prediction")

age = st.number_input("Age", 0, 120)
gender = st.selectbox("Gender", ["Male", "Female"])
country = st.text_input("Country")
diagnosis_date = st.date_input("Diagnosis Date")
cancer_stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
lifestyle = st.selectbox("Lifestyle", ["Active", "Sedentary"])
comorbidities = st.text_input("Comorbidities")
treatment = st.selectbox("Treatment", ["Surgery", "Chemotherapy", "Radiation", "Combination", "None"])

if st.button("Predict Survival"):
    input_df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "country": [country],
        "diagnosis_date": [str(diagnosis_date)],
        "cancer_stage": [cancer_stage],
        "lifestyle": [lifestyle],
        "comorbidities": [comorbidities],
        "treatment": [treatment]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Survival Outcome: {prediction}")



