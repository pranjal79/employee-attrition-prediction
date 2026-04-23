# app.py
# =============================================================================
# PURPOSE: Streamlit web app for Employee Attrition Risk Prediction.
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Fix import paths so `src` is always findable ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── Load secrets BEFORE any other imports that need env vars ────────────────
from dotenv import load_dotenv

def load_secrets():
    """Load from Streamlit secrets (cloud) or .env (local)."""
    try:
        import streamlit as st
        os.environ["MLFLOW_TRACKING_URI"]      = st.secrets["MLFLOW_TRACKING_URI"]
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]
        os.environ["DAGSHUB_USERNAME"]         = st.secrets["DAGSHUB_USERNAME"]
        os.environ["DAGSHUB_TOKEN"]            = st.secrets["DAGSHUB_TOKEN"]
    except Exception:
        load_dotenv()

load_secrets()

# ── Now safe to import everything else ──────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.shap_explain import (
    load_model_and_features,
    build_explainer,
    get_shap_waterfall,
    regenerate_global_shap_plots
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Employee Attrition Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# =============================================================================
# PATHS
# =============================================================================

MODELS_DIR     = os.path.join(BASE_DIR, "models")
SHAP_PLOTS_DIR = os.path.join(BASE_DIR, "shap_plots")

MODEL_PATH     = os.path.join(MODELS_DIR, "model.pkl")
FEATURES_PATH  = os.path.join(MODELS_DIR, "feature_names.pkl")
BAR_PATH       = os.path.join(SHAP_PLOTS_DIR, "shap_summary_bar.png")
BEESWARM_PATH  = os.path.join(SHAP_PLOTS_DIR, "shap_beeswarm.png")

# =============================================================================
# CHECK FILES
# =============================================================================

required = [MODEL_PATH, FEATURES_PATH]
missing  = [f for f in required if not os.path.exists(f)]

if missing:
    st.error("Missing required files:")
    for f in missing:
        st.code(f)
    st.stop()

# =============================================================================
# LOAD MODEL
# =============================================================================

@st.cache_resource
def load_resources():
    model, feature_names = load_model_and_features()
    explainer = build_explainer(model)
    regenerate_global_shap_plots(model=model, explainer=explainer)
    return model, feature_names, explainer

model, feature_names, explainer = load_resources()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    age = st.slider("Age", 18, 60, 35)
    gender = st.selectbox("Gender", ["Female", "Male"])
    gender_encoded = 1 if gender == "Male" else 0
    distance = st.slider("Distance From Home", 1, 29, 5)
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    overtime_encoded = 1 if overtime == "Yes" else 0

# =============================================================================
# INPUT DATA
# =============================================================================

input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_encoded],
    "DistanceFromHome": [distance],
    "OverTime": [overtime_encoded]
})

input_data = input_data.reindex(columns=feature_names, fill_value=0)

# =============================================================================
# PREDICTION
# =============================================================================

prob = model.predict_proba(input_data)[0][1]
pct  = prob * 100

# =============================================================================
# UI
# =============================================================================

st.title("🧠 Employee Attrition Risk Predictor")
st.metric("Attrition Risk", f"{pct:.2f}%")
st.progress(float(prob))

# =============================================================================
# SHAP WATERFALL
# =============================================================================

st.markdown("### 🔍 Explanation")

fig = get_shap_waterfall(input_data, model=model, explainer=explainer)
st.pyplot(fig)
plt.close()

# =============================================================================
# ✅ FIXED GLOBAL SHAP SECTION
# =============================================================================

st.markdown("### 🌍 Global Feature Importance")

tab1, tab2 = st.tabs(["📊 Bar", "🐝 Beeswarm"])

with tab1:
    if os.path.exists(BAR_PATH):
        with open(BAR_PATH, "rb") as f:
            st.image(f.read(), caption="Mean |SHAP Value| per feature", width=900)
    else:
        st.warning(f"Bar plot missing: {BAR_PATH}")

with tab2:
    if os.path.exists(BEESWARM_PATH):
        with open(BEESWARM_PATH, "rb") as f:
            st.image(f.read(), caption="Feature impact direction", width=900)
    else:
        st.warning(f"Beeswarm plot missing: {BEESWARM_PATH}")