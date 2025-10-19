import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import os
import io
import re
import pickle

import mlflow
import dagshub

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------
# DagsHub + MLflow setup
# --------------------------
# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# Set up DagsHub credentials for MLflow tracking
username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")

if not username or not token:
    raise ValueError("Missing DagsHub credentials in environment variables")

# Construct the authenticated MLflow tracking URI
mlflow_uri = f"https://{username}:{token}@dagshub.com/{username}/YouTube-Sentiment-Insights-Plugin.mlflow"

dagshub.init(repo_owner=username, repo_name="YouTube-Sentiment-Insights-Plugin", mlflow=True)
mlflow.set_tracking_uri(mlflow_uri)


app = Flask(__name__)
CORS(app)

# --------------------------
# Preprocessing Function
# --------------------------
def preprocess_comment(comment):
    """Clean and normalize input comment text."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([
            lemmatizer.lemmatize(word)
            for word in comment.split() if word not in stop_words
        ])
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


# --------------------------
# Load Model + Vectorizer
# --------------------------
def load_model_and_vectorizer(model_name):
    """Load staged model and associated vectorizer."""
    try:
        model_uri = f"models:/{model_name}/Staging"
        print(f"🔹 Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        # Load associated vectorizer artifact
        client = mlflow.MlflowClient()
        model_version = client.get_latest_versions(model_name, stages=["Staging"])[0]
        run_id = model_version.run_id
        print(f"🔹 Downloading vectorizer from run: {run_id}")
        vectorizer_local_path = client.download_artifacts(run_id, "tfidf_vectorizer.pkl")

        with open(vectorizer_local_path, 'rb') as f:
            vectorizer = pickle.load(f)

        print("Model and vectorizer loaded successfully from DagsHub (Staging)")
        return model, vectorizer

    except Exception as e:
        print(f"Failed to load staged model: {e}")
        return None, None


# Initialize once at startup
model, vectorizer = load_model_and_vectorizer("youtube-sentiment-lgbm")

# --------------------------
# API Routes
# --------------------------
@app.route('/')
def home():
    return "Welcome to the YouTube Sentiment Insights Flask API 🚀"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess & vectorize
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        feature_names = vectorizer.get_feature_names_out()

        # Convert to DataFrame (critical fix for MLflow schema)
        input_df = pd.DataFrame(transformed.toarray(), columns=feature_names)

        # Predict using pyfunc model
        predictions = model.predict(input_df).tolist()

        response = [
            {"comment": c, "sentiment": int(p)} for c, p in zip(comments, predictions)
        ]
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        feature_names = vectorizer.get_feature_names_out()
        input_df = pd.DataFrame(transformed.toarray(), columns=feature_names)

        predictions = model.predict(input_df).tolist()

        response = [
            {"comment": c, "sentiment": int(p), "timestamp": t}
            for c, p, t in zip(comments, predictions, timestamps)
        ]
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# --------------------------
# Chart, WordCloud, Trend Endpoints
# (unchanged from your working version)
# --------------------------
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=140, textprops={'color': 'w'})
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed_comments = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed_comments)
        wordcloud = WordCloud(
            width=800, height=400, background_color='black',
            colormap='Blues', stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        for sentiment_value in [-1, 0, 1]:
            plt.plot(monthly_percentages.index, monthly_percentages[sentiment_value],
                     marker='o', linestyle='-', label=labels[sentiment_value],
                     color=colors[sentiment_value])

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.tight_layout()
        plt.legend()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": str(e)}), 500


# --------------------------
# App Runner
# --------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
