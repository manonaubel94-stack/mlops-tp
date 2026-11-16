# MLOps TP — DVC + MLflow (DagsHub) + FastAPI + Docker

- **DVC** : `data/raw_dataset.csv` est versionné avec DVC, stocké sur DagsHub (`dvc pull` pour le récupérer).
- **MLflow** : `train.py` entraîne deux RandomForest (dataset digits), log params/métriques sur
  [DagsHub MLflow](https://dagshub.com/manonaubel94-stack/mlops-tp.mlflow) et enregistre le modèle `digits-rf` dans le registry.
- **API** : `app/main.py` expose un seul endpoint `POST /predict`.
- **Docker** : pendant le build, `download_model.py` récupère la dernière version de `digits-rf` depuis DagsHub via MLflow.
- **CI** : `.github/workflows/docker.yml` build l'image et la pousse sur Docker Hub.

## Utilisation

```bash
export MLFLOW_TRACKING_USERNAME=<user dagshub> MLFLOW_TRACKING_PASSWORD=<token dagshub>
python train.py                      # entraîne + enregistre une nouvelle version
DOCKER_BUILDKIT=1 docker build \
  --secret id=mlflow_user,env=MLFLOW_TRACKING_USERNAME \
  --secret id=mlflow_token,env=MLFLOW_TRACKING_PASSWORD -t mlops-tp .
docker run -p 8000:8000 mlops-tp
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"instances": [[0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]]}'
# -> {"predictions":[0],"model_version":"2"}
```

Secrets GitHub requis : `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`.
