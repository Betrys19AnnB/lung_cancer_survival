import gdown
import zipfile
import os
import joblib

# Download the ZIP from Google Drive if not already present
if not os.path.exists("lung_cancer_model.pkl"):
    gdown.download(id="1k20qTxGm1ad_gZb5ev_NlKv4ZMQ6Ko2J", output="lung_cancer_model.zip", quiet=False)

    # Unzip the file
    with zipfile.ZipFile("lung_cancer_model.zip", "r") as zip_ref:
        zip_ref.extractall()

# Load the model
model = joblib.load("lung_cancer_model.pkl")




