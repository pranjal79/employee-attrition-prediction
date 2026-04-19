# src/train.py
# =============================================================================
# PURPOSE: Train 4 ML models, evaluate them, run GridSearchCV on XGBoost,
#          log everything to MLflow + DagsHub, save best model to disk.
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (needed for servers)
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# scikit-learn models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# XGBoost
from xgboost import XGBClassifier

# MLflow experiment tracking
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# SHAP for explainability
import shap

warnings.filterwarnings("ignore")

# =============================================================================
# STEP 1: Load environment variables and configure MLflow → DagsHub
# =============================================================================

load_dotenv()

# Set MLflow authentication using DagsHub credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

# Point MLflow to your DagsHub remote tracking server
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(TRACKING_URI)

print(f"🔗 MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# =============================================================================
# STEP 2: Define all file paths
# =============================================================================

TRAIN_PATH          = "data/processed/train.csv"
TEST_PATH           = "data/processed/test.csv"
FEATURE_NAMES_PATH  = "models/feature_names.pkl"
MODEL_SAVE_PATH     = "models/model.pkl"
SHAP_PLOTS_DIR      = "shap_plots"

os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# =============================================================================
# STEP 3: Load processed data
# =============================================================================

def load_processed_data():
    """Load train and test CSVs saved by preprocess.py."""
    print("\n📂 Loading processed data...")
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    
    # Split into features and target
    X_train = train_df.drop(columns=["Attrition"])
    y_train = train_df["Attrition"]
    X_test  = test_df.drop(columns=["Attrition"])
    y_test  = test_df["Attrition"]
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Train class balance: {y_train.value_counts().to_dict()}")
    print(f"   Test  class balance: {y_test.value_counts().to_dict()}\n")
    
    return X_train, X_test, y_train, y_test

# =============================================================================
# STEP 4: Define the 4 models to train
# =============================================================================

def get_baseline_models():
    """Return dict of model name → model instance."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"   # Handles imbalance at model level too
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0
        )
    }

# =============================================================================
# STEP 5: Evaluate a single model — returns dict of all metrics
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Generate predictions and compute all evaluation metrics.
    Uses predict_proba for ROC-AUC (probability scores).
    """
    # Get class predictions (0 or 1)
    y_pred = model.predict(X_test)
    
    # Get probability of class=1 (attrition) for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "roc_auc"   : round(roc_auc_score(y_test, y_prob), 4),
        "f1_score"  : round(f1_score(y_test, y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred), 4),
        "recall"    : round(recall_score(y_test, y_pred), 4),
    }
    
    print(f"\n📊 {model_name} Results:")
    print(f"   ROC-AUC  : {metrics['roc_auc']}")
    print(f"   F1 Score : {metrics['f1_score']}")
    print(f"   Precision: {metrics['precision']}")
    print(f"   Recall   : {metrics['recall']}")
    
    return metrics

# =============================================================================
# STEP 6: Train all baseline models and log each to MLflow
# =============================================================================

def train_baseline_models(X_train, y_train, X_test, y_test, experiment_name: str):
    """
    Train all 4 baseline models, log params + metrics to MLflow,
    return results dict to find best model.
    """
    mlflow.set_experiment(experiment_name)
    models      = get_baseline_models()
    results     = {}   # model_name → {"model": ..., "metrics": ...}
    
    print("\n" + "="*60)
    print("   TRAINING BASELINE MODELS")
    print("="*60)
    
    for model_name, model in models.items():
        print(f"\n🚀 Training {model_name}...")
        
        # Each model gets its own MLflow run
        with mlflow.start_run(run_name=f"baseline_{model_name}"):
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            metrics = evaluate_model(model, X_test, y_test, model_name)
            
            # Log model hyperparameters to MLflow
            mlflow.log_params(model.get_params())
            
            # Log all evaluation metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # Log the model artifact itself to MLflow
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
            
            # Tag this run for easy filtering
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("stage", "baseline")
        
        results[model_name] = {"model": model, "metrics": metrics}
    
    return results

# =============================================================================
# STEP 7: GridSearchCV on XGBoost (best model)
# Tries all combinations of hyperparameters to find the best configuration
# =============================================================================

def tune_xgboost(X_train, y_train, X_test, y_test, experiment_name: str):
    """
    Run GridSearchCV on XGBoost with cross-validation.
    Logs best params, metrics, and model to MLflow.
    Returns the best fitted model.
    """
    print("\n" + "="*60)
    print("   XGBOOST HYPERPARAMETER TUNING (GridSearchCV)")
    print("="*60)
    
    # Define hyperparameter grid to search
    param_grid = {
        "n_estimators"  : [100, 200],
        "max_depth"     : [3, 5, 7],
        "learning_rate" : [0.01, 0.1, 0.2],
        "subsample"     : [0.8, 1.0],
    }
    
    # Base XGBoost model
    xgb_base = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0
    )
    
    # StratifiedKFold ensures each fold has balanced classes
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # GridSearchCV tries every combination and picks best by ROC-AUC
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring="roc_auc",        # Optimize for AUC
        cv=cv,
        n_jobs=-1,                # Use all CPU cores
        verbose=1
    )
    
    print("🔍 Running grid search (this may take 2-4 minutes)...")
    grid_search.fit(X_train, y_train)
    
    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = round(grid_search.best_score_, 4)
    
    print(f"\n✅ Best Parameters: {best_params}")
    print(f"   Best CV ROC-AUC : {best_cv_score}")
    
    # Evaluate best model on held-out test set
    metrics = evaluate_model(best_model, X_test, y_test, "XGBoost_Tuned")
    metrics["cv_roc_auc"] = best_cv_score
    
    # Log tuned model to MLflow as a separate run
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="XGBoost_GridSearchCV_Best"):
        
        # Log best hyperparameters
        mlflow.log_params(best_params)
        
        # Log all test metrics
        mlflow.log_metrics(metrics)
        
        # Log the tuned model
        mlflow.xgboost.log_model(best_model, artifact_path="best_model")
        
        # Tag as the final best model
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("stage", "tuned_best")
        mlflow.set_tag("tuning_method", "GridSearchCV_5fold")
    
    return best_model, best_params, metrics

# =============================================================================
# STEP 8: Generate and save SHAP plots
# =============================================================================

def generate_shap_plots(model, X_test, feature_names: list):
    """
    Generate SHAP summary plots for global model explainability.
    Saves two PNG files to shap_plots/ directory.
    """
    print("\n" + "="*60)
    print("   GENERATING SHAP EXPLAINABILITY PLOTS")
    print("="*60)
    
    # TreeExplainer is optimized for tree-based models like XGBoost
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # ── Plot 1: Bar chart of mean absolute SHAP values (global importance) ──
    print("\n📊 Generating SHAP summary bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="bar",          # Bar = global importance ranking
        show=False
    )
    plt.title("SHAP Feature Importance (Mean |SHAP Value|)", fontsize=14, pad=15)
    plt.tight_layout()
    bar_path = os.path.join(SHAP_PLOTS_DIR, "shap_summary_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → {bar_path}")
    
    # ── Plot 2: Beeswarm plot (direction + magnitude of each feature) ──
    print("📊 Generating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="dot",          # Dot/beeswarm = direction of effect
        show=False
    )
    plt.title("SHAP Beeswarm — Feature Effects on Attrition", fontsize=14, pad=15)
    plt.tight_layout()
    beeswarm_path = os.path.join(SHAP_PLOTS_DIR, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → {beeswarm_path}")
    
    return bar_path, beeswarm_path

# =============================================================================
# STEP 9: Save best model and feature names to disk
# =============================================================================

def save_best_model(model, feature_names: list):
    """Save best model as model.pkl and feature names as feature_names.pkl."""
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    
    print(f"\n💾 Saved best model    → {MODEL_SAVE_PATH}")
    print(f"💾 Saved feature names → {FEATURE_NAMES_PATH}")

# =============================================================================
# STEP 10: Print final comparison table of all models
# =============================================================================

def print_summary_table(baseline_results: dict, tuned_metrics: dict):
    """Print a clean comparison table of all models."""
    print("\n" + "="*60)
    print("   FINAL MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'ROC-AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-"*60)
    
    for name, result in baseline_results.items():
        m = result["metrics"]
        print(f"{name:<25} {m['roc_auc']:>8} {m['f1_score']:>8} {m['precision']:>10} {m['recall']:>8}")
    
    print(f"{'XGBoost_Tuned':<25} {tuned_metrics['roc_auc']:>8} {tuned_metrics['f1_score']:>8} {tuned_metrics['precision']:>10} {tuned_metrics['recall']:>8}")
    print("="*60)
    print("🏆 Best Model: XGBoost (GridSearchCV Tuned) → saved as model.pkl")

# =============================================================================
# STEP 11: Main pipeline
# =============================================================================

def main():
    print("=" * 60)
    print("   IBM HR ATTRITION — TRAINING PIPELINE")
    print("=" * 60)
    
    EXPERIMENT_NAME = "employee-attrition-experiment"
    
    # Load feature names saved by preprocess.py
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    
    # Load processed train/test data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train all 4 baseline models + log to MLflow
    baseline_results = train_baseline_models(
        X_train, y_train, X_test, y_test, EXPERIMENT_NAME
    )
    
    # Tune XGBoost with GridSearchCV + log best to MLflow
    best_model, best_params, tuned_metrics = tune_xgboost(
        X_train, y_train, X_test, y_test, EXPERIMENT_NAME
    )
    
    # Generate SHAP plots for best model
    bar_path, beeswarm_path = generate_shap_plots(
        best_model, X_test, feature_names
    )
    
    # Log SHAP plots to MLflow (attach to best model run)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="SHAP_Plots_BestModel"):
        mlflow.log_artifact(bar_path)
        mlflow.log_artifact(beeswarm_path)
        mlflow.set_tag("stage", "explainability")
    
    # Save best model + feature names to disk
    save_best_model(best_model, feature_names)
    
    # Print final comparison table
    print_summary_table(baseline_results, tuned_metrics)
    
    print("\n✅ TRAINING PIPELINE COMPLETE!")
    print(f"   Check your DagsHub Experiments tab to see all runs.")
    print(f"   URL: {TRACKING_URI}")


if __name__ == "__main__":
    main()