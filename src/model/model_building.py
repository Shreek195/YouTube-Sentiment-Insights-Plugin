import lightgbm as lgb

import numpy as np
import pandas as pd

import os
import yaml
import pickle
import logging
from src.logger import logging

def load_params(params_path: str) -> dict:
    '''
    Load parameters from yaml file
    '''
    try:
        logging.info("Loading Parameters ....")

        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrived from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not Found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML Error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    '''
    Load data from CSV file
    '''
    try:
        logging.info("Loading Data ....")

        df = pd.read_csv(data_url)
        logging.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Fail to parse the csv file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occured while loading the data: %s', e)
        raise

def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int, num_leaves: int, min_child_samples: int, colsample_bytree: float, subsample: float, reg_alpha: float, reg_lambda: float) -> lgb.LGBMClassifier:
    '''
    Train a LightGBM model
    '''
    try:
        logging.info("Training LightGBM Started ....")

        best_model = lgb.LGBMClassifier(
            # objective='multiclass',
            # num_class=3,
            # metric="multi_logloss",
            # is_unbalance=True,
            # class_weight="balanced",
            reg_alpha=reg_alpha,  # L1 regularization
            reg_lambda=reg_lambda,  # L2 regularization
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            n_estimators=n_estimators,
            # device='gpu',  <- Ubuntu won't have this GPU Support OpenCL
            n_jobs=-1,
            random_state=42
        )
        best_model.fit(X_train, y_train)

        logging.info("Training LightGBM Completed ....")

        logging.debug('LightGBM model training completed')
        return best_model
    except Exception as e:
        logging.error('Error during LightGBM model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    '''
    Save the trained model to a file
    '''
    try:
        logging.info("Saving Model ....")

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.debug('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def get_root_directory() -> str:
    '''
    Get the root directory (two levels up from this script's location)
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'))

        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']
        num_leaves = params['model_building']['num_leaves']
        min_child_samples = params['model_building']['min_child_samples']
        colsample_bytree = params['model_building']['colsample_bytree']
        subsample = params['model_building']['subsample']
        reg_alpha = params['model_building']['reg_alpha']
        reg_lambda = params['model_building']['reg_lambda']

        # Load the preprocessed training data from the interim directory
        train_data = load_data(os.path.join(root_dir, 'data/processed/train_tfidf.csv'))

        X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1]

        # Train the LightGBM model using hyperparameters from params.yaml
        best_model = train_lgbm(X_train, y_train, learning_rate, max_depth, n_estimators, num_leaves, min_child_samples,
        colsample_bytree, subsample, reg_alpha, reg_lambda)

        # Save the trained model in the root directory
        save_model(best_model, os.path.join(root_dir, './models/lgbm_model.pkl'))

    except Exception as e:
        logging.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

