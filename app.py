import gdown
import zipfile
import os
import joblib

# Step 1: Download model zip file
file_id = "1k20qTxGm1ad_gZb5ev_NlKv4ZMQ6Ko2J"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists("lung_cancer_model.pkl"):
    gdown.download(url, "model.zip", quiet=False)
    with zipfile.ZipFile("model.zip", 'r') as zip_ref:
        zip_ref.extractall()

# Step 2: Load the model
model = joblib.load("lung_cancer_model.pkl")



