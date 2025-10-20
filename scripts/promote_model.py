import os
import mlflow
from mlflow import MlflowClient
import dagshub

def setup_mlflow(repo_name: str):
    """
    Set up MLflow tracking URI for DagsHub.
    Works in both local and CI environments.
    """
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    # Fallback for local .env
    if not username or not token:
        from dotenv import load_dotenv
        load_dotenv()
        username = os.getenv("DAGSHUB_USERNAME")
        token = os.getenv("DAGSHUB_TOKEN")

    if not username or not token:
        raise ValueError("Missing DagsHub credentials. Set DAGSHUB_USERNAME and DAGSHUB_TOKEN.")

    # Authenticated MLflow URI
    mlflow_uri = f"https://{username}:{token}@dagshub.com/{username}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(mlflow_uri)

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # ✅ Only initialize dagshub when running locally
    if os.getenv("GITHUB_ACTIONS") != "true":
        dagshub.init(repo_owner=username, repo_name=repo_name, mlflow=True)

    print(f"✅ MLflow tracking URI set to {mlflow_uri}")

def promote_model():
    
    setup_mlflow("YouTube-Sentiment-Insights-Plugin")

    try:
        client = mlflow.MLflowClient()
    except AttributeError:
        client = MlflowClient()

    model_name = "youtube-sentiment-lgbm"
    # Get the latest version 
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        raise ValueError("No model found in Staging stage.")
    latest_version_staging = staging_versions[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=['Production'])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage='Archived'
        )

    # Promote the new model to Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage='Production'
    )

    print(f'Model version {latest_version_staging} promoted to Production')

if __name__ == "__main__":
    # setup_mlflow("YouTube-Sentiment-Insights-Plugin")  # UNCOMMENT WHEN LOCAL RUN

    promote_model()