import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import os
from src.logging.logging import get_logger
logger = get_logger(__name__)


def load_params(params_path: str) -> float:
    try:
        logger.info(f"Loading parameters from {params_path}")

        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        test_size = params['data_ingestion']['test_size']
        logger.debug(f"Test size retrieved: {test_size}")

        return test_size

    except FileNotFoundError:
        logger.error(f"File not found: {params_path}")
        raise

    except KeyError as e:
        logger.error(f"Missing key in YAML: {e}")
        raise

    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {params_path}: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in load_params: {e}")
        raise


def read_data(url: str) -> pd.DataFrame:
    try:
        logger.info(f"Reading data from {url}")

        df = pd.read_csv(url)

        logger.debug(f"Data shape: {df.shape}")
        return df

    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
        raise

    except pd.errors.ParserError:
        logger.error("Error parsing CSV file")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in read_data: {e}")
        raise


def process(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Starting data processing")

        if 'tweet_id' not in df.columns:
            raise KeyError("Column 'tweet_id' not found")

        if 'sentiment' not in df.columns:
            raise KeyError("Column 'sentiment' not found")

        df = df.copy()
        df.drop(columns=['tweet_id'], inplace=True)

        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]

        if final_df.empty:
            raise ValueError("Filtered dataframe is empty")

        final_df['sentiment'] = final_df['sentiment'].replace({
            'happiness': 1,
            'sadness': 0
        })

        logger.debug(f"Processed data shape: {final_df.shape}")

        return final_df

    except KeyError as e:
        logger.error(f"Column error: {e}")
        raise

    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in process: {e}")
        raise


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        logger.info(f"Saving data to {data_path}")

        os.makedirs(data_path, exist_ok=True)

        train_file = os.path.join(data_path, "train.csv")
        test_file = os.path.join(data_path, "test.csv")

        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)

        logger.debug(f"Train data saved: {train_file}")
        logger.debug(f"Test data saved: {test_file}")

    except PermissionError:
        logger.error("Permission denied while saving files")
        raise

    except OSError as e:
        logger.error(f"OS error: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in save_data: {e}")
        raise


def main() -> None:
    try:
        logger.info("Data ingestion pipeline started")

        test_size = load_params('params.yaml')

        df = read_data(
            'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        )

        final_df = process(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        data_path = os.path.join("data", "raw")

        save_data(data_path, train_data, test_data)

        logger.info("Data ingestion completed successfully")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()