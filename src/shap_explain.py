# src/shap_explain.py
# =============================================================================
# PURPOSE: Generate SHAP explanations for the best trained XGBoost model.
#          - Global explanations: summary bar + beeswarm (already done in train.py)
#          - Local explanation: waterfall plot for a SINGLE employee prediction
#          - This file is imported by app.py to explain individual predictions
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — works on servers & Streamlit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# STEP 1: Define file paths
# =============================================================================

MODEL_PATH         = "models/model.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"
SHAP_PLOTS_DIR     = "shap_plots"

os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)


# =============================================================================
# STEP 2: Load model and feature names
# These are cached at module level so they load only ONCE when app.py imports
# this file — not on every user interaction (keeps the app fast)
# =============================================================================

def load_model_and_features():
    """
    Load the saved XGBoost model and feature names from disk.
    Returns model and feature_names list.
    """
    print("📦 Loading model and feature names...")
    model         = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print(f"   ✅ Model loaded: {type(model).__name__}")
    print(f"   ✅ Features ({len(feature_names)}): {feature_names}\n")
    return model, feature_names


# =============================================================================
# STEP 3: Build the SHAP TreeExplainer
# TreeExplainer is specifically optimized for tree-based models (XGBoost,
# RandomForest, GradientBoosting). It's much faster than KernelExplainer.
# =============================================================================

def build_explainer(model):
    """
    Create a SHAP TreeExplainer for the given model.
    This computes SHAP values using the fast tree path algorithm.
    """
    print("🔍 Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    print("   ✅ Explainer ready.\n")
    return explainer


# =============================================================================
# STEP 4: Get SHAP waterfall chart for ONE employee
# This is the KEY function called by app.py on every prediction.
# A waterfall chart shows:
#   - Starting baseline (average model output across all employees)
#   - Each feature's contribution: pushing prediction UP (red) or DOWN (blue)
#   - Final predicted probability for THIS specific employee
# =============================================================================

def get_shap_waterfall(input_df: pd.DataFrame, model=None, explainer=None) -> plt.Figure:
    """
    Generate a SHAP waterfall plot for a single employee input.
    
    Parameters:
    -----------
    input_df  : pd.DataFrame — single row with all feature columns
    model     : loaded XGBoost model (optional, loads from disk if None)
    explainer : SHAP TreeExplainer (optional, builds if None)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure — waterfall chart to display in Streamlit
    """
    # Load model and build explainer if not passed in
    if model is None:
        model, _ = load_model_and_features()
    if explainer is None:
        explainer = build_explainer(model)
    
    # Compute SHAP values for this single employee row
    # shap_values shape: (1, n_features) — one row, one value per feature
    shap_values = explainer.shap_values(input_df)
    
    # Get the expected value (baseline — average prediction across training data)
    expected_value = explainer.expected_value
    
    # shap_values[0] = SHAP values for the first (only) row
    shap_vals_single = shap_values[0]
    
    # Build a SHAP Explanation object (needed for waterfall plot API)
    explanation = shap.Explanation(
        values        = shap_vals_single,           # SHAP values for each feature
        base_values   = expected_value,             # Baseline prediction
        data          = input_df.values[0],         # Actual feature values
        feature_names = input_df.columns.tolist()   # Feature names for labels
    )
    
    # ── Draw the waterfall plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # shap.plots.waterfall renders into the current matplotlib figure
    shap.plots.waterfall(
        explanation,
        max_display=15,    # Show top 15 most impactful features
        show=False         # Don't call plt.show() — we return the figure instead
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
# STEP 5: Regenerate global SHAP plots (bar + beeswarm) from saved test data
# This function is useful if you want to re-run SHAP plots without retraining.
# =============================================================================

def regenerate_global_shap_plots(model=None, explainer=None):
    """
    Regenerate SHAP summary bar and beeswarm plots using the test set.
    Saves PNGs to shap_plots/ directory.
    Called once during app startup to ensure plots always exist.
    """
    test_path = "data/processed/test.csv"
    
    # Skip if plots already exist (avoid recomputing every app restart)
    bar_path      = os.path.join(SHAP_PLOTS_DIR, "shap_summary_bar.png")
    beeswarm_path = os.path.join(SHAP_PLOTS_DIR, "shap_beeswarm.png")
    
    if os.path.exists(bar_path) and os.path.exists(beeswarm_path):
        print("✅ Global SHAP plots already exist — skipping regeneration.")
        return bar_path, beeswarm_path
    
    print("📊 Regenerating global SHAP plots...")
    
    # Load model and features if not passed in
    if model is None:
        model, feature_names = load_model_and_features()
    else:
        feature_names = joblib.load(FEATURE_NAMES_PATH)
    
    if explainer is None:
        explainer = build_explainer(model)
    
    # Load test data for global explanation
    test_df = pd.read_csv(test_path)
    X_test  = test_df.drop(columns=["Attrition"])
    
    # Compute SHAP values for all test rows
    shap_values = explainer.shap_values(X_test)
    
    # ── Bar plot: Mean absolute SHAP value per feature ──────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance (Global)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → {bar_path}")
    
    # ── Beeswarm plot: Direction + magnitude of each feature ─────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="dot",
        show=False
    )
    plt.title("SHAP Beeswarm — Direction of Feature Effects", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved → {beeswarm_path}")
    
    return bar_path, beeswarm_path


# =============================================================================
# STEP 6: Quick test — run this file directly to verify everything works
# =============================================================================

def run_test():
    """
    Standalone test: load model, build explainer, generate waterfall for
    a fake employee row, and save the plot to shap_plots/test_waterfall.png
    """
    print("=" * 60)
    print("   SHAP EXPLAIN — STANDALONE TEST")
    print("=" * 60 + "\n")
    
    # Load model and features
    model, feature_names = load_model_and_features()
    
    # Build SHAP explainer
    explainer = build_explainer(model)
    
    # Create a fake employee row using median/mode values
    # (In the real app, this comes from the Streamlit sidebar inputs)
    fake_employee = pd.DataFrame(
        columns=feature_names,
        data=[[
            35,   # Age
            1,    # BusinessTravel (1=Travel_Rarely)
            800,  # DailyRate
            1,    # Department (1=Research & Development)
            5,    # DistanceFromHome
            3,    # Education
            2,    # EducationField
            3,    # EnvironmentSatisfaction
            1,    # Gender (1=Male)
            5000, # HourlyRate
            0,    # JobInvolvement
            3,    # JobLevel
            5,    # JobRole
            3,    # JobSatisfaction
            1,    # MaritalStatus
            5000, # MonthlyIncome
            15000,# MonthlyRate
            2,    # NumCompaniesWorked
            0,    # OverTime (0=No)
            15,   # PercentSalaryHike
            3,    # PerformanceRating
            3,    # RelationshipSatisfaction
            80,   # StockOptionLevel — actually 0-3, set to 0
            5,    # TotalWorkingYears
            3,    # TrainingTimesLastYear
            3,    # WorkLifeBalance
            5,    # YearsAtCompany
            3,    # YearsInCurrentRole
            2,    # YearsSinceLastPromotion
            3,    # YearsWithCurrManager
        ]]
    )
    
    # Align fake employee columns to exact model feature order
    fake_employee = fake_employee[feature_names]
    
    print("🧪 Generating waterfall for fake employee row...")
    fig = get_shap_waterfall(fake_employee, model=model, explainer=explainer)
    
    # Save test waterfall plot
    test_path = os.path.join(SHAP_PLOTS_DIR, "test_waterfall.png")
    fig.savefig(test_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"   ✅ Test waterfall saved → {test_path}")
    
    # Also regenerate global plots
    regenerate_global_shap_plots(model=model, explainer=explainer)
    
    print("\n✅ SHAP EXPLAIN TEST COMPLETE!")
    print(f"   Check shap_plots/ for: test_waterfall.png, shap_summary_bar.png, shap_beeswarm.png")


if __name__ == "__main__":
    run_test()