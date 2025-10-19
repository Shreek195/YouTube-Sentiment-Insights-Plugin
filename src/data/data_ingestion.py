from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import os
import yaml
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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocess the data by handling missing values, duplicates, and empty strings
    '''
    try:
        logging.info("Data Preprocessing ....")

        # Removing the missing values
        df.dropna(inplace=True)

        # Removing Duplicates
        df.drop_duplicates(inplace=True)

        # Removing empty rows
        df = df[df['clean_comment'].str.strip() != '']

        logging.debug('Data preprocessing completed: Missing values, duplicates and empty rows removed')
        return df
    except KeyError as e:
        logging.error('Missing column error in dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    '''
    Save the train and test datasets, creating the raw folder if it doesn't exist
    '''
    try:
        logging.info("Saving the train test data ....")

        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occured while saving the data: %s', e)
        raise

def main():
    try:
        # Load parameters from params.yaml in the root directory
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']

        # Load data from specific url
        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

        # Preprocess the data
        final_df = preprocess_data(df)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()