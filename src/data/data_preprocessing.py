import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Tuple
from src.logging.logging import get_logger

logger = get_logger(__name__)

# ===================== NLTK =====================
nltk.download('wordnet')
nltk.download('stopwords')


# ===================== LOAD DATA =====================
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


# ===================== TEXT CLEANING =====================
def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    except Exception as e:
        logger.exception(f"Error in lemmatization: {e}")
        raise


def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        words = [w for w in str(text).split() if w not in stop_words]
        return " ".join(words)

    except Exception as e:
        logger.exception(f"Error in remove_stop_words: {e}")
        raise


def removing_numbers(text: str) -> str:
    try:
        return ''.join([c for c in text if not c.isdigit()])

    except Exception as e:
        logger.exception(f"Error in removing_numbers: {e}")
        raise


def lower_case(text: str) -> str:
    try:
        return " ".join([w.lower() for w in text.split()])

    except Exception as e:
        logger.exception(f"Error in lower_case: {e}")
        raise


def removing_punctuations(text: str) -> str:
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = re.sub('\s+', ' ', text)
        return text.strip()

    except Exception as e:
        logger.exception(f"Error in removing_punctuations: {e}")
        raise


def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)

    except Exception as e:
        logger.exception(f"Error in removing_urls: {e}")
        raise


# ===================== NORMALIZATION =====================
def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Starting text normalization")

        if 'content' not in df.columns:
            raise KeyError("Column 'content' not found")

        df = df.copy()

        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)

        logger.debug(f"Normalized dataframe shape: {df.shape}")
        return df

    except KeyError as e:
        logger.error(f"Column error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Error in normalize_text: {e}")
        raise


# ===================== PROCESS =====================
def processed_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    try:
        logger.info("Processing train and test data")

        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)

        return train_processed, test_processed

    except Exception as e:
        logger.exception(f"Error in processed_data: {e}")
        raise


# ===================== SAVE =====================
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    try:
        logger.info("Saving processed data")

        data_path = os.path.join("data", "processed")
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train_processed.csv")
        test_path = os.path.join(data_path, "test_processed.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.debug(f"Saved files: {train_path}, {test_path}")

    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        raise

    except OSError as e:
        logger.error(f"OS error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in save_data: {e}")
        raise


# ===================== MAIN =====================
def main() -> None:
    try:
        logger.info("Data preprocessing pipeline started")

        train_path = os.path.join("data", "raw", "train.csv")
        test_path = os.path.join("data", "raw", "test.csv")

        train_data, test_data = load_data(train_path, test_path)

        train_processed, test_processed = processed_data(train_data, test_data)

        save_data(train_processed, test_processed)

        logger.info("Data preprocessing completed successfully")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()