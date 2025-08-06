import os
import pandas as pd
import joblib
import gdown
from flask import Flask, request, render_template

app = Flask(__name__)

# === Load the model from Google Drive if not present ===
model_path = "lung_cancer_model.pkl"
gdrive_id = "1k20qTxGm1ad_gZb5ev_NlKv4ZMQ6Ko2J"
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", model_path, quiet=False)

# === Load model ===
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = [features]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'Survival Prediction Score: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
