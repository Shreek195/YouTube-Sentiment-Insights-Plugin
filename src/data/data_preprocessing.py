import numpy as np
import pandas as pd

import os
import yaml
import logging
from src.logger import logging

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_comment(comment):
    '''
    Apply preprocessing transformations to a comment
    '''
    try:
        # logging.info("Comments Preprocessing ....")

        # Convert to lower
        comment = comment.lower()

        # Removing Noise
        comment = re.sub(r"http\S+|www\S+|@\S+|#\S+|[^a-zA-Z\s]", "", comment)
        comment = re.sub(r"\s+", " ", comment).strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logging.error(f"Error in preprocessing comment: {e}")
        return comment

def normalize_text(df):
    '''
    Apply preprocessing to the text data in the dataframe
    '''
    try:
        logging.info("Normalize Comments ....")

        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)

        # Removing the missing values
        df.dropna(inplace=True)
        # Removing Duplicates
        df.drop_duplicates(inplace=True)

        logging.debug('Text normalization completed')
        return df
    except Exception as e:
        logging.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    '''
    Save the processed train and test datasets
    '''
    try:
        logging.info("Save train test data ....")

        interim_data_path = os.path.join(data_path, 'interim')
        logging.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logging.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logging.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logging.debug("Starting data preprocessing...")
        
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.debug('Data loaded successfully')

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the processed data
        save_data(train_processed_data, test_processed_data, data_path='./data')
    except Exception as e:
        logging.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()