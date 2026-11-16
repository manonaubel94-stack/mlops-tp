"""Entraîne deux RandomForest (comme MLFlowSetup.ipynb), log sur DagsHub/MLflow
et enregistre chaque modèle dans le Model Registry sous MODEL_NAME.

Variables d'environnement requises :
    MLFLOW_TRACKING_USERNAME -> utilisateur DagsHub
    MLFLOW_TRACKING_PASSWORD -> token DagsHub
"""
import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "manonaubel94-stack")
REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "mlops-tp")
EXPERIMENT_NAME = "RandomForest_V1.0.1"
MODEL_NAME = os.getenv("MODEL_NAME", "digits-rf")

# Équivalent de dagshub.init(..., mlflow=True) : pointe MLflow vers le serveur DagsHub
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")
mlflow.set_experiment(EXPERIMENT_NAME)

data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)


def train_and_log_model(n_estimators, max_depth, run_name):
    with mlflow.start_run(run_name=run_name):
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            registered_model_name=MODEL_NAME,  # enregistre une nouvelle version
        )
        print(
            f"{run_name}: n_estimators={n_estimators}, max_depth={max_depth}, "
            f"accuracy={accuracy:.4f}"
        )


if __name__ == "__main__":
    print("Training and Logging Model 1...")
    train_and_log_model(n_estimators=20, max_depth=5, run_name="random_forest_model_run_1")
    print("\nTraining and Logging Model 2...")
    train_and_log_model(n_estimators=100, max_depth=10, run_name="random_forest_model_run_V1.0.2")
