from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import ADASYN

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

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    '''
    Apply TF-IDF with ngrams to the data
    '''
    try:
        logging.info("Initializing Vectorizer ....")

        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values
        
        X_test = test_data['clean_comment'].values
        y_test = test_data['category'].values

        # Perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_tfidf = pd.DataFrame(X_train_tfidf.toarray())
        train_tfidf['label'] = y_train

        test_tfidf = pd.DataFrame(X_test_tfidf.toarray())
        test_tfidf['label'] = y_test

        logging.debug(f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}")

        # Save the vectorizer in the root directory
        with open(os.path.join(get_root_directory(), 'models/tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logging.debug('TF-IDF applied with trigrams and data transformed')
        return train_tfidf, test_tfidf
    except Exception as e:
        logging.error('Error during TF-IDF transformation: %s', e)
        raise

def save_data(df: pd.DataFrame, data_path: str) -> None:
    '''
    Save the processed train and test datasets
    '''
    try:
        logging.info("Save train test data ....")
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)  # Ensure the directory is created
        logging.debug(f"Directory {data_path} created or already exists")

        df.to_csv(data_path, index=False)
        
        logging.debug(f"Processed data saved to {data_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving data: {e}")
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
        max_features = params['feature_engineering']['max_features']
        ngram_range = tuple(params['feature_engineering']['ngram_range'])

        # Load the preprocessed training data from the interim directory
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))
        test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

        # Removing the missing values
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)
        # Removing Duplicates
        train_data.drop_duplicates(inplace=True)
        test_data.drop_duplicates(inplace=True)

        logging.info(f"Is there na values train: {train_data.isna().sum()}, test: {test_data.isna().sum()}")

        # Apply TF-IDF feature engineering on training data
        train_tfidf, test_tfidf = apply_tfidf(train_data, test_data, max_features, ngram_range)

        save_data(train_tfidf, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_tfidf, os.path.join("./data", "processed", "test_tfidf.csv"))    

    except Exception as e:
        logging.error('Failed to complete the feature engineering %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()