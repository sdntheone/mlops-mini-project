import numpy as np
import pandas as pd
import pickle
import os
import yaml
import json

from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin

import mlflow
import mlflow.sklearn
import dagshub

from mlflow.models import infer_signature  # ✅ IMPORTANT

from src.logging.logging import get_logger

logger = get_logger(__name__)


# -----------------------------
# Load params
# -----------------------------
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)['model_building']


# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri('https://dagshub.com/sdntheone/mlops-mini-project.mlflow')
dagshub.init(repo_owner='sdntheone', repo_name='mlops-mini-project', mlflow=True)


# -----------------------------
# Load data
# -----------------------------
def load_data(train_data_path: str) -> pd.DataFrame:
    logger.info(f"Loading training data from: {train_data_path}")
    return pd.read_csv(train_data_path)


# -----------------------------
# Split
# -----------------------------
def split_x_y(train_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("Splitting data")
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    return X_train, y_train


# -----------------------------
# Build model
# -----------------------------
def build_model(X_train: np.ndarray, y_train: np.ndarray) -> ClassifierMixin:
    logger.info("Training Logistic Regression model")

    clf = LogisticRegression(
        C=params['C'],
        solver=params['solver'],
        penalty=params['penalty']
    )

    clf.fit(X_train, y_train)
    return clf


# -----------------------------
# Save model (DVC)
# -----------------------------
def save_model(clf: ClassifierMixin, path: str):
    logger.info(f"Saving model to {path}")

    with open(path, 'wb') as f:
        pickle.dump(clf, f)


# -----------------------------
# MAIN
# -----------------------------
def main():
    try:
        logger.info("Model building started")

        base_dir = os.getcwd()

        train_path = os.path.join(base_dir, "data", "features", "train_tfidf.csv")

        train_data = load_data(train_path)
        X_train, y_train = split_x_y(train_data)

        clf = build_model(X_train, y_train)

        # -----------------------------
        # Save model (DVC requirement)
        # -----------------------------
        model_dir = os.path.join(base_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.pkl")
        save_model(clf, model_path)

        logger.info(f"Model saved at {model_path}")

        # -----------------------------
        # MLflow logging
        # -----------------------------
        mlflow.set_experiment("dvc-pipeline")

        with mlflow.start_run() as run:

            # log params
            for k, v in params.items():
                mlflow.log_param(k, v)

            # ✅ FIX: proper MLflow model logging
            signature = infer_signature(X_train, clf.predict(X_train))

            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                signature=signature,
                input_example=X_train[:2]
            )

            # -----------------------------
            # Save model info for registry
            # -----------------------------
            reports_dir = os.path.join(base_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)

            model_info_path = os.path.join(reports_dir, "model_info.json")

            with open(model_info_path, "w") as f:
                json.dump({
                    "run_id": run.info.run_id,
                    "artifact_path": "model",
                    "model_uri": f"runs:/{run.info.run_id}/model"
                }, f, indent=4)

            logger.info(f"Model info saved at {model_info_path}")

        logger.info("Model building completed successfully")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()