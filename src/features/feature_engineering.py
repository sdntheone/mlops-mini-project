import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
from typing import Tuple
from scipy.sparse import csr_matrix
from src.logging.logging import get_logger

logger = get_logger(__name__)


with open('params.yaml', 'r') as f:
    max_features: int = yaml.safe_load(f)['feature_engineering']['max_features']


def load_data(train_data_path: str, test_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.info(f"Loading data from: {train_data_path}, {test_data_path}")

        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)

        logger.debug(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        return train_data, test_data

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


def fillna(train_data_path: str, test_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.info("Filling missing values")

        train_data, test_data = load_data(train_data_path, test_data_path)

        train_data = train_data.fillna('')
        test_data = test_data.fillna('')

        return train_data, test_data

    except Exception as e:
        logger.exception(f"Error in fillna: {e}")
        raise


def convert_to_numpy(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    try:
        logger.info("Converting data to numpy arrays")

        if 'content' not in train_data.columns or 'sentiment' not in train_data.columns:
            raise KeyError("Required columns missing in train_data")

        if 'content' not in test_data.columns or 'sentiment' not in test_data.columns:
            raise KeyError("Required columns missing in test_data")

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        logger.debug(f"X_train: {len(X_train)}, X_test: {len(X_test)}")

        return X_train, y_train, X_test, y_test

    except KeyError as e:
        logger.error(f"Column error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in convert_to_numpy: {e}")
        raise


def bag_of_words(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[csr_matrix, csr_matrix]:

    try:
        logger.info("Applying Tfidf")

        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        logger.debug(f"Tfidf shapes: {X_train_tfidf.shape}, {X_test_tfidf.shape}")

        return X_train_tfidf, X_test_tfidf

    except ValueError as e:
        logger.error(f"Vectorization error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in bag_of_words: {e}")
        raise


def tfidf_to_df(
    X_train_tfidf: csr_matrix,
    X_test_tfidf: csr_matrix,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    try:
        logger.info("Converting sparse matrix to DataFrame")

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        return train_df, test_df

    except ValueError as e:
        logger.error(f"DataFrame conversion error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in bow_to_df: {e}")
        raise


def save_to_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    try:
        logger.info("Saving feature data")

        data_path = os.path.join("data", "features")
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train_tfidf.csv")
        test_path = os.path.join(data_path, "test_tfidf.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.debug(f"Saved: {train_path}, {test_path}")

    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        raise

    except OSError as e:
        logger.error(f"OS error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in save_to_features: {e}")
        raise


def main() -> None:
    try:
        logger.info("Feature engineering pipeline started")

        train_data_path = os.path.join("data", "processed", "train_processed.csv")
        test_data_path = os.path.join("data", "processed", "test_processed.csv")

        train_data, test_data = fillna(train_data_path, test_data_path)

        X_train, y_train, X_test, y_test = convert_to_numpy(train_data, test_data)

        X_train_bow, X_test_bow = bag_of_words(X_train, X_test)

        train_df, test_df = tfidf_to_df(X_train_bow, X_test_bow, y_train, y_test)

        save_to_features(train_df, test_df)

        logger.info("Feature engineering completed successfully")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()