"""Télécharge la dernière version du modèle depuis le Model Registry MLflow de DagsHub.
Exécuté pendant le `docker build`.

Variables d'environnement : MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
"""
import os

import mlflow

REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "manonaubel94-stack")
REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "mlops-tp")
MODEL_NAME = os.getenv("MODEL_NAME", "digits-rf")
OUTPUT_DIR = os.getenv("MODEL_DIR", "model")

mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")
client = mlflow.MlflowClient()

latest = max(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))
print(f"Téléchargement de {MODEL_NAME} v{latest.version} (run {latest.run_id})")
mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{MODEL_NAME}/{latest.version}", dst_path=OUTPUT_DIR
)
with open(os.path.join(OUTPUT_DIR, "VERSION"), "w") as f:
    f.write(latest.version)
