# src/shap_explain.py
# =============================================================================
# PURPOSE: Generate SHAP explanations for the best trained XGBoost model
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# ✅ FIXED PATHS — WORKS LOCALLY + STREAMLIT CLOUD
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH         = os.path.join(BASE_DIR, "models", "model.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")
SHAP_PLOTS_DIR     = os.path.join(BASE_DIR, "shap_plots")

os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

# =============================================================================
# LOAD MODEL + FEATURES
# =============================================================================

def load_model_and_features():
    print("📦 Loading model and feature names...")
    model         = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print(f"   ✅ Model loaded: {type(model).__name__}")
    print(f"   ✅ Features ({len(feature_names)})")
    return model, feature_names

# =============================================================================
# BUILD EXPLAINER
# =============================================================================

def build_explainer(model):
    print("🔍 Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    print("   ✅ Explainer ready.")
    return explainer

# =============================================================================
# WATERFALL (LOCAL EXPLANATION)
# =============================================================================

def get_shap_waterfall(input_df: pd.DataFrame, model=None, explainer=None):
    if model is None:
        model, _ = load_model_and_features()
    if explainer is None:
        explainer = build_explainer(model)

    shap_values = explainer.shap_values(input_df)
    expected_value = explainer.expected_value
    shap_vals_single = shap_values[0]

    explanation = shap.Explanation(
        values=shap_vals_single,
        base_values=expected_value,
        data=input_df.values[0],
        feature_names=input_df.columns.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False
    )

    plt.title(
        "SHAP Waterfall — Why This Employee Is At Risk",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    plt.tight_layout()

    return fig

# =============================================================================
# GLOBAL SHAP PLOTS
# =============================================================================

def regenerate_global_shap_plots(model=None, explainer=None):

    bar_path      = os.path.join(SHAP_PLOTS_DIR, "shap_summary_bar.png")
    beeswarm_path = os.path.join(SHAP_PLOTS_DIR, "shap_beeswarm.png")

    # Skip if already exists
    if os.path.exists(bar_path) and os.path.exists(beeswarm_path):
        print("✅ SHAP plots already exist.")
        return bar_path, beeswarm_path

    print("📊 Generating global SHAP plots...")

    if model is None:
        model, feature_names = load_model_and_features()
    else:
        feature_names = joblib.load(FEATURE_NAMES_PATH)

    if explainer is None:
        explainer = build_explainer(model)

    # ⚠️ NOTE: You must have test.csv committed OR this will fail
    test_path = os.path.join(BASE_DIR, "data", "processed", "test.csv")

    if not os.path.exists(test_path):
        print("⚠️ test.csv not found → skipping SHAP global plots")
        return None, None

    test_df = pd.read_csv(test_path)
    X_test  = test_df.drop(columns=["Attrition"])

    shap_values = explainer.shap_values(X_test)

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance (Global)")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    plt.close()

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="dot",
        show=False
    )
    plt.title("SHAP Beeswarm")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()

    print("✅ SHAP plots saved.")

    return bar_path, beeswarm_path

# =============================================================================
# TEST FUNCTION
# =============================================================================

def run_test():
    print("Running SHAP test...")

    model, feature_names = load_model_and_features()
    explainer = build_explainer(model)

    fake = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

    fig = get_shap_waterfall(fake, model, explainer)

    test_path = os.path.join(SHAP_PLOTS_DIR, "test_waterfall.png")
    fig.savefig(test_path)
    plt.close()

    print(f"✅ Test plot saved → {test_path}")


if __name__ == "__main__":
    run_test()