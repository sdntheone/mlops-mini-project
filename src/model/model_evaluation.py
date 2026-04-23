from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.base import ClassifierMixin
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import pickle
import json
import os

import mlflow
import dagshub

from src.logging.logging import get_logger

logger = get_logger(__name__)

# mlflow.set_tracking_uri('https://dagshub.com/sdntheone/mlops-mini-project.mlflow')
# dagshub.init(repo_owner='sdntheone', repo_name='mlops-mini-project', mlflow=True)

dagshub_token=os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT envoiroment variable is not set ")

os.environ["MLFLOW_TRACKING_USERNAME"]=dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"]=dagshub_token

dagshub_url="https://dagshub.com"
repo_owner="sdntheone"
repo_name="mlops-mini-project"

# setup MLFlow tracking URL
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


def load_model(path: str) -> ClassifierMixin:
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_test_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    test_data = pd.read_csv(data_path)
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    return X_test, y_test


def prediction(clf: ClassifierMixin, X_test: np.ndarray):
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba


def evaluation_metrics(y_test, y_pred, y_pred_proba) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }


def save_metrics(metrics_dict: Dict[str, float], path: str):
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)


def main():

    mlflow.set_experiment("dvc-pipeline")

    with mlflow.start_run():

        try:
            logger.info("Starting evaluation pipeline")

            # 🔥 Use absolute paths (fixes DVC + MLflow issues)
            base_dir = os.getcwd()

            model_path = os.path.join(base_dir, "models", "model.pkl")
            test_data_path = os.path.join(base_dir, "data", "features", "test_tfidf.csv")

            reports_dir = os.path.join(base_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)

            metrics_path = os.path.join(reports_dir, "metrics.json")

            # Load
            clf = load_model(model_path)
            X_test, y_test = load_test_data(test_data_path)

            # Predict
            y_pred, y_pred_proba = prediction(clf, X_test)

            # Metrics
            metrics = evaluation_metrics(y_test, y_pred, y_pred_proba)

            # Save metrics
            save_metrics(metrics, metrics_path)

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log params
            if hasattr(clf, "get_params"):
                for k, v in clf.get_params().items():
                    mlflow.log_param(k, v)

            # 🔥 IMPORTANT: DO NOT log model here
            # mlflow.sklearn.log_model(clf, "model")  ❌ REMOVED

            # Log artifacts
            mlflow.log_artifact(metrics_path)

            if os.path.exists("model_evaluation_errors.log"):
                mlflow.log_artifact("model_evaluation_errors.log")

            logger.info("Pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            print(f"Error: {e}")
            raise


if __name__ == "__main__":
    main()