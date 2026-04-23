# app.py
# =============================================================================
# PURPOSE: Streamlit web app for Employee Attrition Risk Prediction
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================

import os
from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# ABSOLUTE PATHS (FIX FOR STREAMLIT CLOUD)
# =============================================================================

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
bar_path      = os.path.join(BASE_DIR, "shap_plots", "shap_summary_bar.png")
beeswarm_path = os.path.join(BASE_DIR, "shap_plots", "shap_beeswarm.png")

# =============================================================================
# SECRETS LOADER
# =============================================================================

def load_secrets():
    try:
        os.environ["MLFLOW_TRACKING_URI"]      = st.secrets["MLFLOW_TRACKING_URI"]
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]
        os.environ["DAGSHUB_USERNAME"]         = st.secrets["DAGSHUB_USERNAME"]
        os.environ["DAGSHUB_TOKEN"]            = st.secrets["DAGSHUB_TOKEN"]
    except Exception:
        load_dotenv()

load_secrets()

# =============================================================================
# IMPORT CUSTOM MODULES
# =============================================================================

from src.data_loader import check_required_files
from src.shap_explain import (
    load_model_and_features,
    build_explainer,
    get_shap_waterfall,
    regenerate_global_shap_plots
)

# =============================================================================
# STREAMLIT CONFIG
# =============================================================================

st.set_page_config(
    page_title="Employee Attrition Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ✅ CHECK FILES FIRST (CRITICAL FIX)
# =============================================================================

if not check_required_files():
    st.stop()

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
.main { background-color: #f8f9fa; }

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 5px solid #dee2e6;
}
.metric-card h2 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.metric-card p {
    font-size: 0.85rem;
    color: #6c757d;
}

[data-testid="stSidebar"] { background-color: #1e2130; }
[data-testid="stSidebar"] * { color: #e9ecef !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL (CACHED)
# =============================================================================

@st.cache_resource(show_spinner="Loading model and explainer...")
def load_resources():
    model, feature_names = load_model_and_features()
    explainer = build_explainer(model)
    regenerate_global_shap_plots(model=model, explainer=explainer)
    return model, feature_names, explainer

model, feature_names, explainer = load_resources()

# =============================================================================
# SIDEBAR INPUTS
# =============================================================================

with st.sidebar:
    st.markdown("## 👤 Employee Profile")

    age = st.slider("Age", 18, 60, 35)
    gender = st.selectbox("Gender", ["Female", "Male"])
    gender_encoded = 1 if gender == "Male" else 0

    distance_from_home = st.slider("Distance From Home", 1, 29, 5)

    overtime = st.selectbox("OverTime", ["No", "Yes"])
    overtime_encoded = 1 if overtime == "Yes" else 0

# =============================================================================
# INPUT DATA
# =============================================================================

input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_encoded],
    "DistanceFromHome": [distance_from_home],
    "OverTime": [overtime_encoded]
})

input_data = input_data.reindex(columns=feature_names, fill_value=0)

# =============================================================================
# PREDICTION
# =============================================================================

attrition_prob = model.predict_proba(input_data)[0][1]
attrition_pct = attrition_prob * 100

# =============================================================================
# UI OUTPUT
# =============================================================================

st.title("🧠 Employee Attrition Risk Predictor")

st.metric("Attrition Risk", f"{attrition_pct:.2f}%")

st.progress(float(attrition_prob))

# =============================================================================
# SHAP WATERFALL
# =============================================================================

st.markdown("### 🔍 Explanation")

fig = get_shap_waterfall(input_data, model=model, explainer=explainer)
st.pyplot(fig)
plt.close()

# =============================================================================
# GLOBAL SHAP PLOTS
# =============================================================================

st.markdown("### 🌍 Global Feature Importance")

col1, col2 = st.columns(2)

with col1:
    if os.path.exists(bar_path):
        st.image(bar_path, caption="Feature Importance (Bar)", use_container_width=True)
    else:
        st.warning("Bar plot not found")

with col2:
    if os.path.exists(beeswarm_path):
        st.image(beeswarm_path, caption="Beeswarm Plot", use_container_width=True)
    else:
        st.warning("Beeswarm plot not found")