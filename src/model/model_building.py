import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import yaml
from typing import Tuple
from sklearn.base import ClassifierMixin
from src.logging.logging import get_logger

logger = get_logger(__name__)


with open('params.yaml','r') as f:
    params = yaml.safe_load(f)['model_building']


def load_data(train_data_path: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading training data from: {train_data_path}")

        train_data = pd.read_csv(train_data_path)

        logger.debug(f"Train data shape: {train_data.shape}")
        return train_data

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty CSV file: {e}")
        raise

    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in load_data: {e}")
        raise


def split_x_y(train_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    try:
        logger.info("Splitting data into features and target")

        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values

        logger.debug(f"X shape: {X_train.shape}, y shape: {y_train.shape}")
        return X_train, y_train

    except Exception as e:
        logger.exception(f"Error in split_x_y: {e}")
        raise


def build_model(X_train: np.ndarray, y_train: np.ndarray) -> ClassifierMixin:
    try:
        logger.info("Building Logistic Regression model")

        clf = LogisticRegression(
        C=params['C'],
        solver=params['solver'],
        penalty=params['penalty']
            )

        clf.fit(X_train, y_train)

        logger.info("Model training completed")
        return clf

    except ValueError as e:
        logger.error(f"Model training error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in build_model: {e}")
        raise


def save_model(clf: ClassifierMixin, path: str) -> None:
    try:
        logger.info(f"Saving model to: {path}")

        with open(path, 'wb') as f:
            pickle.dump(clf, f)

        logger.debug("Model saved successfully")

    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        raise

    except OSError as e:
        logger.error(f"OS error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in save_model: {e}")
        raise


def main() -> None:
    try:
        logger.info("Model building pipeline started")

        train_data_path = os.path.join("data", "features", "train_tfidf.csv")

        train_data = load_data(train_data_path)

        X_train, y_train = split_x_y(train_data)

        clf = build_model(X_train, y_train)

        save_model(clf, 'model.pkl')

        logger.info("Model building completed successfully")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()