import numpy as np
import pandas as pd

import pickle
import logging
from src.logger import logging
import yaml
import json

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.models import infer_signature

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import os

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub

# Load environment variables  DO THIS WHEN LOCAL
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


def load_params(params_path: str) -> dict:
    '''
    Load parameters from yaml file
    '''
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    '''
    Load data from a CSV file
    '''
    try:
        df = pd.read_csv(file_path)
        logging.debug('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logging.error('Error loading data from %s: %s', file_path, e)
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    '''
    Load the saved TF-IDF vectorizer
    '''
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logging.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logging.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise

def load_model(model_path: str):
    '''
    Load the trained model
    '''
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logging.error('Error loading model from %s: %s', model_path, e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    '''
    Evaluate the model and log classification metrics and confusion matrix
    '''
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logging.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name):
    '''
    Log confusion matrix as an artifact or figure
    '''
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    mlflow.log_figure(plt.gcf(), "Confusion_Matrix.png")
    plt.close()

def save_metrics(metrics: dict, file_path: str) -> None:
    '''
    Save the evaluation metrics to a JSON file
    '''
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    '''
    Save the model run ID and path to a JSON file
    '''
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment('dvc-pipeline-runs-v1')

    with mlflow.start_run() as run:
        # ------------------ Metadata ------------------
        mlflow.set_tag("mlflow.runName", "LGBM_TFIDF_TriGram_ADASYN")
        mlflow.set_tag("experiment_stage", "model_evaluation")
        # mlflow.set_tag("experiment_type", "hypertuning")
        mlflow.set_tag("model_type", "LightGBMClassifier")
        mlflow.set_tag("description", "LightGBM model evaluated with TF-IDF TriGram features and ADASYN resampling on YouTube comments dataset.")
        mlflow.set_tag("task", "Sentiment Analysis")
        mlflow.set_tag("dataset", "YouTube Comments")

        try:
            # Load parameters from YAML file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, './models/lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, './models/tfidf_vectorizer.pkl'))

            # Load test data
            test_data = load_data(os.path.join(root_dir, 'data/processed/test_tfidf.csv'))

            # Prepare test data
            X_test_tfidf = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Convert to float64 to handle missing values
            X_test_tfidf = X_test_tfidf.astype('float64')
            
            # Create DataFrame with feature names
            feature_names = vectorizer.get_feature_names_out()
            X_test_df = pd.DataFrame(X_test_tfidf, columns=feature_names)
            
            # Handle any missing values
            if X_test_df.isnull().any().any():
                logging.warning("Missing values detected in test data. Filling with 0.")
                X_test_df.fillna(0, inplace=True)

            # Create input example for MLflow (first 5 rows)
            input_example = X_test_df.head(5)

            # Infer signature with proper data types
            # Make predictions on input example
            example_predictions = model.predict(input_example)
            signature = infer_signature(input_example, example_predictions)

            # Log model with signature
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="lgbm_model",
                signature=signature,  # Now includes proper float64 schema
                input_example=input_example,
            )

            # Save model info
            # artifact_uri = mlflow.get_artifact_uri()
            # model_path = f"{artifact_uri}/lgbm_model"
            save_model_info(run.info.run_id, 'model', './reports/experiment_info.json') 

            # Log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, './models/tfidf_vectorizer.pkl'))

            # Evaluate with DataFrame (preserves feature names)
            report, cm = evaluate_model(model, X_test_df, y_test)
            save_metrics(report, 'reports/metrics.json')

            # Log classification report metrics
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log overall accuracy
            if 'accuracy' in report:
                mlflow.log_metric("test_accuracy", report['accuracy'])

            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            logging.info("Model evaluation completed successfully")

        except Exception as e:
            logging.error(f"Failed to complete model evaluation: {e}")
            raise

if __name__ == '__main__':
    main()