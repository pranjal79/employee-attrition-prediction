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
# STEP 1: Streamlit page config (must be FIRST streamlit command)
# =============================================================================

st.set_page_config(
    page_title="Employee Attrition Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STEP 2: Custom CSS
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
.metric-card h2 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
}
.metric-card p {
    font-size: 0.85rem;
    color: #6c757d;
    margin: 4px 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.card-red    { border-left-color: #dc3545; }
.card-red    h2 { color: #dc3545; }
.card-orange { border-left-color: #fd7e14; }
.card-orange h2 { color: #fd7e14; }
.card-green  { border-left-color: #28a745; }
.card-green  h2 { color: #28a745; }
.card-blue   { border-left-color: #007bff; }
.card-blue   h2 { color: #007bff; }

.risk-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 50px;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-top: 6px;
}
.badge-high   { background: #dc3545; color: white; }
.badge-medium { background: #fd7e14; color: white; }
.badge-low    { background: #28a745; color: white; }

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #343a40;
    border-bottom: 2px solid #e9ecef;
    padding-bottom: 6px;
    margin: 24px 0 16px 0;
}

[data-testid="stSidebar"] { background-color: #1e2130; }
[data-testid="stSidebar"] * { color: #e9ecef !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# STEP 3: Absolute paths — works on local AND Streamlit Cloud
# =============================================================================

MODELS_DIR     = os.path.join(BASE_DIR, "models")
SHAP_PLOTS_DIR = os.path.join(BASE_DIR, "shap_plots")
MODEL_PATH     = os.path.join(MODELS_DIR, "model.pkl")
FEATURES_PATH  = os.path.join(MODELS_DIR, "feature_names.pkl")
BAR_PATH       = os.path.join(SHAP_PLOTS_DIR, "shap_summary_bar.png")
BEESWARM_PATH  = os.path.join(SHAP_PLOTS_DIR, "shap_beeswarm.png")


# =============================================================================
# STEP 4: Verify required files exist before loading
# =============================================================================

required = [MODEL_PATH, FEATURES_PATH]
missing  = [f for f in required if not os.path.exists(f)]

if missing:
    st.error("⚠️ Required model files are missing. Please run the training pipeline first.")
    for f in missing:
        st.code(f"Missing: {f}")
    st.info("Run locally:\n```\npython src/preprocess.py\npython src/train.py\n```")
    st.stop()


# =============================================================================
# STEP 5: Load model + explainer — cached so loads only ONCE
# =============================================================================

@st.cache_resource(show_spinner="Loading model and explainer...")
def load_resources():
    """Load model + build SHAP explainer once and cache."""
    model, feature_names = load_model_and_features()
    explainer            = build_explainer(model)
    regenerate_global_shap_plots(model=model, explainer=explainer)
    return model, feature_names, explainer


model, feature_names, explainer = load_resources()


# =============================================================================
# STEP 6: Sidebar inputs
# =============================================================================

with st.sidebar:
    st.markdown("## 👤 Employee Profile")
    st.markdown("Adjust the sliders and dropdowns to match the employee's details.")
    st.markdown("---")

    # ── Personal ──────────────────────────────────────────────────────────────
    st.markdown("### 🧍 Personal")

    age = st.slider("Age", 18, 60, 35)

    gender = st.selectbox("Gender", ["Female", "Male"], index=1)
    gender_encoded = 1 if gender == "Male" else 0

    marital_status  = st.selectbox(
        "Marital Status", ["Divorced", "Married", "Single"], index=1
    )
    marital_encoded = ["Divorced", "Married", "Single"].index(marital_status)

    distance_from_home = st.slider("Distance From Home (km)", 1, 29, 5)

    st.markdown("---")

    # ── Job Details ───────────────────────────────────────────────────────────
    st.markdown("### 💼 Job Details")

    department   = st.selectbox(
        "Department",
        ["Human Resources", "Research & Development", "Sales"],
        index=1
    )
    dept_encoded = ["Human Resources", "Research & Development", "Sales"].index(department)

    job_role_list = [
        "Healthcare Representative", "Human Resources", "Laboratory Technician",
        "Manager", "Manufacturing Director", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative"
    ]
    job_role         = st.selectbox("Job Role", job_role_list, index=6)
    job_role_encoded = job_role_list.index(job_role)

    job_level = st.slider("Job Level", 1, 5, 2)

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        ["1 — Low", "2 — Medium", "3 — High", "4 — Very High"], index=2
    )
    job_sat_encoded = int(job_satisfaction[0])

    job_involvement = st.selectbox(
        "Job Involvement",
        ["1 — Low", "2 — Medium", "3 — High", "4 — Very High"], index=2
    )
    job_inv_encoded = int(job_involvement[0])

    overtime         = st.selectbox("OverTime", ["No", "Yes"], index=0)
    overtime_encoded = 1 if overtime == "Yes" else 0

    business_travel = st.selectbox(
        "Business Travel",
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=1
    )
    travel_encoded = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(business_travel)

    st.markdown("---")

    # ── Compensation ──────────────────────────────────────────────────────────
    st.markdown("### 💰 Compensation")

    monthly_income      = st.slider("Monthly Income ($)", 1000, 20000, 5000, step=100)
    daily_rate          = st.slider("Daily Rate", 100, 1500, 800, step=10)
    hourly_rate         = st.slider("Hourly Rate", 30, 100, 65)
    monthly_rate        = st.slider("Monthly Rate", 2000, 27000, 14000, step=100)
    percent_salary_hike = st.slider("Percent Salary Hike (%)", 11, 25, 14)

    stock_option_level = st.selectbox(
        "Stock Option Level",
        ["0 — None", "1 — Low", "2 — Medium", "3 — High"], index=1
    )
    stock_encoded = int(stock_option_level[0])

    st.markdown("---")

    # ── Experience & Tenure ───────────────────────────────────────────────────
    st.markdown("### 📅 Experience & Tenure")

    total_working_years        = st.slider("Total Working Years", 0, 40, 8)
    years_at_company           = st.slider("Years At Company", 0, 40, 5)
    years_in_current_role      = st.slider("Years In Current Role", 0, 18, 3)
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    years_with_curr_manager    = st.slider("Years With Current Manager", 0, 17, 4)
    num_companies_worked       = st.slider("Num Companies Worked", 0, 9, 2)
    training_times_last_year   = st.slider("Training Times Last Year", 0, 6, 3)

    st.markdown("---")

    # ── Satisfaction & Balance ────────────────────────────────────────────────
    st.markdown("### 😊 Satisfaction & Balance")

    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        ["1 — Low", "2 — Medium", "3 — High", "4 — Very High"], index=2
    )
    env_sat_encoded = int(environment_satisfaction[0])

    relationship_satisfaction = st.selectbox(
        "Relationship Satisfaction",
        ["1 — Low", "2 — Medium", "3 — High", "4 — Very High"], index=2
    )
    rel_sat_encoded = int(relationship_satisfaction[0])

    work_life_balance = st.selectbox(
        "Work Life Balance",
        ["1 — Bad", "2 — Good", "3 — Better", "4 — Best"], index=2
    )
    wlb_encoded = int(work_life_balance[0])

    st.markdown("---")

    # ── Education ─────────────────────────────────────────────────────────────
    st.markdown("### 🎓 Education")

    education = st.selectbox(
        "Education Level",
        ["1 — Below College", "2 — College", "3 — Bachelor", "4 — Master", "5 — Doctor"],
        index=2
    )
    edu_encoded = int(education[0])

    edu_field_list  = [
        "Human Resources", "Life Sciences", "Marketing",
        "Medical", "Other", "Technical Degree"
    ]
    education_field  = st.selectbox("Education Field", edu_field_list, index=1)
    edu_field_encoded = edu_field_list.index(education_field)

    performance_rating = st.selectbox(
        "Performance Rating",
        ["1 — Low", "2 — Good", "3 — Excellent", "4 — Outstanding"], index=2
    )
    perf_encoded = int(performance_rating[0])


# =============================================================================
# STEP 7: Build input DataFrame in exact feature order the model expects
# =============================================================================

input_data = pd.DataFrame([{
    "Age"                      : age,
    "BusinessTravel"           : travel_encoded,
    "DailyRate"                : daily_rate,
    "Department"               : dept_encoded,
    "DistanceFromHome"         : distance_from_home,
    "Education"                : edu_encoded,
    "EducationField"           : edu_field_encoded,
    "EnvironmentSatisfaction"  : env_sat_encoded,
    "Gender"                   : gender_encoded,
    "HourlyRate"               : hourly_rate,
    "JobInvolvement"           : job_inv_encoded,
    "JobLevel"                 : job_level,
    "JobRole"                  : job_role_encoded,
    "JobSatisfaction"          : job_sat_encoded,
    "MaritalStatus"            : marital_encoded,
    "MonthlyIncome"            : monthly_income,
    "MonthlyRate"              : monthly_rate,
    "NumCompaniesWorked"       : num_companies_worked,
    "OverTime"                 : overtime_encoded,
    "PercentSalaryHike"        : percent_salary_hike,
    "PerformanceRating"        : perf_encoded,
    "RelationshipSatisfaction" : rel_sat_encoded,
    "StockOptionLevel"         : stock_encoded,
    "TotalWorkingYears"        : total_working_years,
    "TrainingTimesLastYear"    : training_times_last_year,
    "WorkLifeBalance"          : wlb_encoded,
    "YearsAtCompany"           : years_at_company,
    "YearsInCurrentRole"       : years_in_current_role,
    "YearsSinceLastPromotion"  : years_since_last_promotion,
    "YearsWithCurrManager"     : years_with_curr_manager,
}])

# Reorder columns to exactly match model training feature order
input_data = input_data[feature_names]


# =============================================================================
# STEP 8: Run prediction
# =============================================================================

attrition_prob = model.predict_proba(input_data)[0][1]
retention_prob = 1 - attrition_prob
attrition_pct  = attrition_prob * 100
retention_pct  = retention_prob * 100

if attrition_pct < 35:
    risk_level = "Low"
    badge_class = "badge-low"
    card_class  = "card-green"
    risk_emoji  = "🟢"
elif attrition_pct < 60:
    risk_level  = "Medium"
    badge_class = "badge-medium"
    card_class  = "card-orange"
    risk_emoji  = "🟠"
else:
    risk_level  = "High"
    badge_class = "badge-high"
    card_class  = "card-red"
    risk_emoji  = "🔴"


# =============================================================================
# STEP 9: Main panel — page header
# =============================================================================

st.markdown("# 🧠 Employee Attrition Risk Predictor")
st.markdown(
    "Powered by **XGBoost + SHAP** · Adjust the sidebar to profile an employee "
    "and instantly see their attrition risk with AI-driven explanations."
)
st.markdown("---")


# =============================================================================
# STEP 10: Three metric cards
# =============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    color = (
        "card-red"    if risk_level == "High"   else
        "card-orange" if risk_level == "Medium" else
        "card-green"
    )
    st.markdown(f"""
    <div class="metric-card {color}">
        <h2>{attrition_pct:.1f}%</h2>
        <p>⚠️ Attrition Risk</p>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card card-blue">
        <h2 style="font-size:1.5rem;">{risk_emoji} {risk_level}</h2>
        <span class="risk-badge {badge_class}">{risk_level.upper()} RISK</span>
        <p style="margin-top:8px;">Risk Level</p>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card card-green">
        <h2>{retention_pct:.1f}%</h2>
        <p>✅ Retention Probability</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# STEP 11: Risk progress bar
# =============================================================================

st.markdown('<div class="section-header">📊 Risk Gauge</div>', unsafe_allow_html=True)
st.markdown(f"**{risk_emoji} {risk_level} Risk — {attrition_pct:.1f}% probability of attrition**")
st.progress(float(attrition_prob))
st.caption("🟢 Low: 0–35%  |  🟠 Medium: 35–60%  |  🔴 High: >60%")
st.markdown("---")


# =============================================================================
# STEP 12: Employee input summary (collapsible)
# =============================================================================

with st.expander("📋 View Current Employee Input Data", expanded=False):
    st.dataframe(
        pd.DataFrame({
            "Feature" : feature_names,
            "Value"   : input_data.values[0]
        }),
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# STEP 13: SHAP Waterfall chart — explains THIS employee's prediction
# =============================================================================

st.markdown(
    '<div class="section-header">🔍 Why Is This Employee At Risk? (SHAP Waterfall)</div>',
    unsafe_allow_html=True
)
st.markdown(
    "Each bar shows how much a feature **increases** 🔴 or **decreases** 🔵 "
    "the attrition risk for **this specific employee**."
)

with st.spinner("Generating SHAP explanation..."):
    waterfall_fig = get_shap_waterfall(input_data, model=model, explainer=explainer)
    st.pyplot(waterfall_fig, use_container_width=True)
    plt.close()

st.markdown("---")


# =============================================================================
# STEP 14: Global SHAP summary plots (bar + beeswarm) in tabs
# =============================================================================

st.markdown(
    '<div class="section-header">🌍 Global Feature Importance (All Employees)</div>',
    unsafe_allow_html=True
)
st.markdown("These plots show which features matter most **across the entire dataset**.")

tab1, tab2 = st.tabs(["📊 Feature Importance (Bar)", "🐝 Beeswarm (Direction of Effect)"])

with tab1:
    if os.path.exists(BAR_PATH):
        with open(BAR_PATH, "rb") as f:
            st.image(
                f.read(),
                caption="Mean |SHAP Value| per feature — higher = more important",
                width=900
            )
    else:
        st.warning(f"Bar plot not found at: {BAR_PATH}")
        st.info("Run `python src/train.py` locally and push shap_plots/ to GitHub.")

with tab2:
    if os.path.exists(BEESWARM_PATH):
        with open(BEESWARM_PATH, "rb") as f:
            st.image(
                f.read(),
                caption="Red = pushes toward attrition · Blue = pushes toward retention",
                width=900
            )
    else:
        st.warning(f"Beeswarm plot not found at: {BEESWARM_PATH}")
        st.info("Run `python src/train.py` locally and push shap_plots/ to GitHub.")

st.markdown("---")


# =============================================================================
# STEP 15: Footer
# =============================================================================

st.markdown("""
<div style="text-align:center; color:#6c757d; font-size:0.8rem; padding:12px 0;">
    🧠 Employee Attrition Risk Predictor &nbsp;·&nbsp;
    Built with XGBoost + SHAP + Streamlit &nbsp;·&nbsp;
    Tracked with MLflow on DagsHub
</div>
""", unsafe_allow_html=True)