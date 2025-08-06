import os
import pickle
import zipfile
import gdown
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_ZIP_NAME = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\lung_cancer_survival\\models\\lung_cancer_model.zip"
MODEL_FILE_NAME = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\lung_cancer_survival\\models\\lung_cancer_model.pkl"
GOOGLE_DRIVE_FILE_ID = "1k20qTxGm1ad_gZb5ev_NlKv4ZMQ6Ko2J"

# Download and extract the model if not already present
def download_model():
    if not os.path.exists(MODEL_FILE_NAME):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_ZIP_NAME, quiet=False)

        print("Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall()
        print("Model extracted.")

download_model()

# Load the model
with open(MODEL_FILE_NAME, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return "Lung Cancer Survival Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
