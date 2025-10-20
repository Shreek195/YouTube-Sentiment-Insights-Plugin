import os
import io
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import mlflow
import dagshub
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------
# MLflow setup helper
# --------------------------
def setup_mlflow(repo_name: str):
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if not username or not token:
        from dotenv import load_dotenv
        load_dotenv()
        username = os.getenv("DAGSHUB_USERNAME")
        token = os.getenv("DAGSHUB_TOKEN")

    if not username or not token:
        raise ValueError("Missing DagsHub credentials")

    mlflow_uri = f"https://{username}:{token}@dagshub.com/{username}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(mlflow_uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    dagshub.init(repo_owner=username, repo_name=repo_name, mlflow=True)
    print(f"✅ MLflow tracking URI set to {mlflow_uri}")

# --------------------------
# Preprocessing
# --------------------------
def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'no', 'however', 'yet'}
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in comment.split() if word not in stop_words])
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# --------------------------
# Model Loader
# --------------------------
def load_model_and_vectorizer(model_name):
    """Load staged model and vectorizer. Returns (None, None) if failed."""
    try:
        model_uri = f"models:/{model_name}/Staging"
        print(f"🔹 Loading model from {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        client = mlflow.MlflowClient()
        version_info = client.get_latest_versions(model_name, stages=["Staging"])[0]
        run_id = version_info.run_id
        vectorizer_path = client.download_artifacts(run_id, "tfidf_vectorizer.pkl")

        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        print("✅ Model and vectorizer loaded successfully")
        return model, vectorizer

    except Exception as e:
        print(f"❌ Failed to load model/vectorizer: {e}")
        return None, None

# --------------------------
# App Factory (MLOps-friendly)
# --------------------------
def create_app(testing=False):
    """
    Flask app factory.
    If `testing=True`, MLflow/model loading is skipped for CI/unit tests.
    """
    app = Flask(__name__)
    CORS(app)

    model, vectorizer = (None, None)
    if not testing:
        try:
            setup_mlflow("YouTube-Sentiment-Insights-Plugin")
            model, vectorizer = load_model_and_vectorizer("youtube-sentiment-lgbm")
        except Exception as e:
            print(f"Warning: Skipping model loading: {e}")

    @app.route('/')
    def home():
        return "<title>Sentiment Analysis</title><h1>Welcome</h1>"

    @app.route('/predict', methods=['POST'])
    def predict():
        if testing:
            # Dummy response for tests
            return jsonify([{"comment": c, "sentiment": 1} for c in request.json.get("comments", [])])

        if model is None or vectorizer is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.json
        comments = data.get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        input_df = pd.DataFrame(transformed.toarray(), columns=vectorizer.get_feature_names_out())
        predictions = model.predict(input_df).tolist()

        response = [{"comment": c, "sentiment": int(p)} for c, p in zip(comments, predictions)]
        return jsonify(response)

    return app

# --------------------------
# Global app for running locally
# --------------------------
app = create_app()

# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
