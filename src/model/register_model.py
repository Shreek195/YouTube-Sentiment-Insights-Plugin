import mlflow

import os
import pickle
import yaml
import json
import logging
from src.logger import logging

import dagshub

# Load environment variables DO THIS WHEN LOCAL
# from dotenv import load_dotenv
# load_dotenv()



# Set up DagsHub credentials for MLflow tracking
username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")

if not username or not token:
    raise ValueError("Missing DagsHub credentials in environment variables")

# Construct the authenticated MLflow tracking URI
mlflow_uri = f"https://{username}:{token}@dagshub.com/{username}/YouTube-Sentiment-Insights-Plugin.mlflow"

dagshub.init(repo_owner=username, repo_name="YouTube-Sentiment-Insights-Plugin", mlflow=True)
mlflow.set_tracking_uri(mlflow_uri)


def load_model_info(file_path: str) -> dict:
    '''
    Load the model info from a JSON file
    '''
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, run_id: str, artifact_path: str) -> dict:
    '''
    Register the model to the MLflow Model Registry using run-based URI
    
    Args:
        model_name: Name for the registered model
        run_id: MLflow run ID where model was logged
        artifact_path: Path to model artifact within the run (default: "lgbm_model")
    
    Returns:
        dict with registration info
    '''
    try:
        client = mlflow.MlflowClient()
        
        # Construct run-based URI for registration
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        logging.info(f"Registering model '{model_name}' from: {model_uri}")

        # Register the model - this creates the models:/ reference
        model_version = mlflow.register_model(model_uri, model_name)
        
        version = model_version.version
        logging.info(f'Model registered: {model_name} version {version}')

        # Transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False
        )

        # Now you can use: models:/{model_name}/{version} or models:/{model_name}/Staging
        models_uri_version = f"models:/{model_name}/{version}"
        models_uri_stage = f"models:/{model_name}/Staging"
        
        # Save the model URIs for later use
        registration_info = {
            "model_name": model_name,
            "version": version,
            "stage": "Staging",
            "run_id": run_id,
            "models_uri_version": models_uri_version,
            "models_uri_stage": models_uri_stage
        }
        
        # Save to file
        with open('./reports/model_registry_info.json', 'w') as f:
            json.dump(registration_info, f, indent=4)
        
        logging.info("Model registration info saved to ./reports/model_registry_info.json")
        
        return registration_info
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        run_id = model_info['run_id']
        model_name = "youtube-sentiment-lgbm"
        # register_model(model_name, model_info)

        registration_info = register_model(
            model_name=model_name,
            run_id=run_id,
            artifact_path="lgbm_model"
        )

        print(f"\nYou can now load the model using:")
        print(f"  model = mlflow.pyfunc.load_model('{registration_info['models_uri_stage']}')")
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()