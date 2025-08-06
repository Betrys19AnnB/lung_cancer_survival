import gdown
import joblib
import os

# Download model only if not already downloaded
if not os.path.exists("lung_cancer_model.pkl"):
    gdown.download(id="1k20qTxGm1ad_gZb5ev_NlKv4ZMQ6Ko2J", output="lung_cancer_model.tar.gz", quiet=False)
    import tarfile
    with tarfile.open("lung_cancer_model.tar.gz", "r:gz") as tar:
        tar.extractall()

# Load model
model = joblib.load("lung_cancer_model.pkl")





