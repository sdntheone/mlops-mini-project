import json
import mlflow
import logging
import os
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
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}/{repo_name}.mlflow')


def load_model_info(file_path:str) -> dict:
    """ load the model info from json"""
    try:
        with open(file_path,'r') as file:
            model_info=json.load(file)
        logger.debug("Model info loaded from %s",file_path)
        return model_info
    
    except FileNotFoundError:
        logger.error('File not found: %s',file_path)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the model info: %s",e)
        raise

def register_model(model_name: str, model_info: dict):
    try:
        run_id = model_info["run_id"]

        # ✅ Use model URI with explicit model artifact
        model_uri = f"runs:/{run_id}/model"

        print("DEBUG → MODEL URI:", model_uri)

        # 🔥 KEY FIX: use create_registered_model + create version
        client = mlflow.tracking.MlflowClient()

        # create model if not exists
        try:
            client.create_registered_model(model_name)
        except:
            pass

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        print(f"✅ Model {model_name} v{model_version.version} registered successfully")

    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise
    
def main():
    try:
          model_info_path="reports/model_info.json"
          model_info=load_model_info(model_info_path)

          model_name="my_model"
          register_model(model_name,model_info)
    except Exception as e:
         logger.error("Failed to complete the model registration process: %s",e)
         print(f"Error:{e}")


if __name__=="__main__":
     main()

