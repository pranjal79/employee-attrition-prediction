# setup_env.py
# Run this once to verify DagsHub + MLflow connection works

import os
from dotenv import load_dotenv
import mlflow

# Load all variables from .env file into environment
load_dotenv()

# Read each variable
dagshub_user = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_user = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_pass = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Print confirmation (never print the actual token in real code)
print(f"✅ DAGSHUB_USERNAME     : {dagshub_user}")
print(f"✅ DAGSHUB_TOKEN        : {'*' * len(dagshub_token) if dagshub_token else 'NOT SET'}")
print(f"✅ MLFLOW_TRACKING_URI  : {tracking_uri}")

# Set MLflow credentials as environment variables for authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_user
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_pass

# Point MLflow to DagsHub
mlflow.set_tracking_uri(tracking_uri)

print(f"\n🔗 MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")

# Try creating a test experiment to confirm connection
try:
    mlflow.set_experiment("connection-test")
    with mlflow.start_run(run_name="test-connection"):
        mlflow.log_param("test_param", "hello")
        mlflow.log_metric("test_metric", 1.0)
    print("\n✅ MLflow connection to DagsHub SUCCESSFUL!")
    print("👉 Check your DagsHub repo → Experiments tab to see the test run.")
except Exception as e:
    print(f"\n❌ Connection FAILED: {e}")
    print("👉 Double-check your DAGSHUB_TOKEN and MLFLOW_TRACKING_URI in .env")