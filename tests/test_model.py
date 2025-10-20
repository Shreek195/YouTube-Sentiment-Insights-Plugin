import unittest
import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------
# Preprocessing function (same as API)
# --------------------------
def preprocess_comment(comment):
    comment = comment.lower().strip()
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in comment.split() if w not in stop_words])


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup DagsHub MLflow credentials
        username = os.getenv("DAGSHUB_USERNAME")
        token = os.getenv("DAGSHUB_TOKEN")
        if not token:
            raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        repo_name = "YouTube-Sentiment-Insights-Plugin"
        mlflow_uri = f"https://{username}:{token}@dagshub.com/{username}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(mlflow_uri)

        # Model name
        cls.model_name = "youtube-sentiment-lgbm"

        try:
            # Load model from Staging
            cls.model_uri = f"models:/{cls.model_name}/Staging"
            cls.model = mlflow.pyfunc.load_model(cls.model_uri)

            # Load vectorizer artifact
            client = mlflow.MlflowClient()
            model_version = client.get_latest_versions(cls.model_name, stages=["Staging"])
            if not model_version:
                raise unittest.SkipTest(f"Model {cls.model_name} not in Staging")
            run_id = model_version[0].run_id
            vectorizer_path = client.download_artifacts(run_id, "tfidf_vectorizer.pkl")
            with open(vectorizer_path, "rb") as f:
                cls.vectorizer = pickle.load(f)

            # Load holdout data
            cls.holdout_data = pd.read_csv("data/processed/test_tfidf.csv")

        except Exception as e:
            raise unittest.SkipTest(f"Skipping tests: failed to load model/vectorizer ({e})")

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        # Example input (same preprocessing as API)
        input_text = "This is a sample comment for testing!"
        preprocessed = preprocess_comment(input_text)
        transformed = self.vectorizer.transform([preprocessed])
        feature_names = self.vectorizer.get_feature_names_out()
        input_df = pd.DataFrame(transformed.toarray(), columns=feature_names)

        # Predict
        prediction = self.model.predict(input_df)
        self.assertEqual(len(prediction), 1)
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, 0:-1].copy()
        y_holdout = self.holdout_data.iloc[:, -1]

        # Match vectorizer columns
        X_holdout.columns = self.vectorizer.get_feature_names_out()

        y_pred = self.model.predict(X_holdout)

        accuracy_new = accuracy_score(y_holdout, y_pred)
        precision_new = precision_score(y_holdout, y_pred, average='weighted')
        recall_new = recall_score(y_holdout, y_pred, average='weighted')
        f1_new = f1_score(y_holdout, y_pred, average='weighted')

        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        self.assertGreaterEqual(accuracy_new, expected_accuracy)
        self.assertGreaterEqual(precision_new, expected_precision)
        self.assertGreaterEqual(recall_new, expected_recall)
        self.assertGreaterEqual(f1_new, expected_f1)



if __name__ == "__main__":
    unittest.main()
