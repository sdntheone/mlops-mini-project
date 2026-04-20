from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.base import ClassifierMixin
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import pickle
import json
import os

import mlflow
import mlflow.sklearn
import dagshub

from src.logging.logging import get_logger

logger = get_logger(__name__)

# -----------------------------
# MLflow + DagsHub setup
# -----------------------------
mlflow.set_tracking_uri('https://dagshub.com/sdntheone/mlops-mini-project.mlflow')
dagshub.init(repo_owner='sdntheone', repo_name='mlops-mini-project', mlflow=True)


# -----------------------------
# Load model
# -----------------------------
def load_model(path: str) -> ClassifierMixin:
    with open(path, 'rb') as f:
        return pickle.load(f)


# -----------------------------
# Load test data
# -----------------------------
def load_test_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    test_data = pd.read_csv(data_path)
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    return X_test, y_test


# -----------------------------
# Prediction
# -----------------------------
def prediction(clf: ClassifierMixin, X_test: np.ndarray):
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba


# -----------------------------
# Metrics
# -----------------------------
def evaluation_metrics(y_test, y_pred, y_pred_proba) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }


# -----------------------------
# Save metrics locally
# -----------------------------
def save_metrics(metrics_dict: Dict[str, float], path: str):
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)


# -----------------------------
# Save run info
# -----------------------------
def save_model_info(run_id: str, artifact_path: str, path: str):
    info = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_uri": f"runs:/{run_id}/{artifact_path}"
    }
    with open(path, 'w') as f:
        json.dump(info, f, indent=4)


# -----------------------------
# MAIN (UPDATED)
# -----------------------------
def main():

    mlflow.set_experiment("dvc-pipeline")

    with mlflow.start_run() as run:

        try:
            logger.info("Starting evaluation pipeline")

            model_path = 'model.pkl'
            test_data_path = os.path.join("data", "features", "test_tfidf.csv")

            # -----------------------------
            # Load
            # -----------------------------
            clf = load_model(model_path)
            X_test, y_test = load_test_data(test_data_path)

            # -----------------------------
            # Predict
            # -----------------------------
            y_pred, y_pred_proba = prediction(clf, X_test)

            # -----------------------------
            # Metrics
            # -----------------------------
            metrics = evaluation_metrics(y_test, y_pred, y_pred_proba)

            save_metrics(metrics, 'metrics.json')

            # -----------------------------
            # Log metrics
            # -----------------------------
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # -----------------------------
            # Log params
            # -----------------------------
            if hasattr(clf, "get_params"):
                params = clf.get_params()
                for k, v in params.items():
                    mlflow.log_param(k, v)

            # -----------------------------
            # Log model
            # -----------------------------
            mlflow.sklearn.log_model(clf, "model")

            # -----------------------------
            # Save run info
            # -----------------------------
            save_model_info(
                run.info.run_id,
                "model",
                "reports/model_info.json"
            )

            # -----------------------------
            # Log artifacts
            # -----------------------------
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('reports/model_info.json')

            # optional log file if exists
            if os.path.exists("model_evaluation_errors.log"):
                mlflow.log_artifact("model_evaluation_errors.log")

            logger.info("Pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            print(f"Error: {e}")
            raise


if __name__ == "__main__":
    main()