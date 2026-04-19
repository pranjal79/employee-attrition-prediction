# app.py
# =============================================================================
# PURPOSE: Streamlit web app for Employee Attrition Risk Prediction.
#          Users input employee details via sidebar → model predicts attrition
#          risk → SHAP waterfall explains WHY → global SHAP shows overall trends.
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Import our custom SHAP module
from src.shap_explain import (
    load_model_and_features,
    build_explainer,
    get_shap_waterfall,
    regenerate_global_shap_plots
)

# =============================================================================
# STEP 1: Streamlit page configuration (must be FIRST streamlit command)
# =============================================================================

st.set_page_config(
    page_title = "Employee Attrition Risk Predictor",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# =============================================================================
# STEP 2: Custom CSS styling for metric cards and risk badges
# =============================================================================

st.markdown("""
<style>
/* ── Main background ── */
.main { background-color: #f8f9fa; }

/* ── Metric card base style ── */
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

/* ── Coloured card variants ── */
.card-red   { border-left-color: #dc3545; }
.card-red   h2 { color: #dc3545; }

.card-orange { border-left-color: #fd7e14; }
.card-orange h2 { color: #fd7e14; }

.card-green { border-left-color: #28a745; }
.card-green h2 { color: #28a745; }

.card-blue  { border-left-color: #007bff; }
.card-blue  h2 { color: #007bff; }

/* ── Risk badge ── */
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

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #343a40;
    border-bottom: 2px solid #e9ecef;
    padding-bottom: 6px;
    margin: 24px 0 16px 0;
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background-color: #1e2130;
}
[data-testid="stSidebar"] * {
    color: #e9ecef !important;
}
[data-testid="stSidebar"] .stSlider label {
    color: #adb5bd !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# STEP 3: Load model, features, and explainer — cached so they load ONCE
# st.cache_resource keeps heavy objects (model, explainer) in memory
# across all user sessions — critical for app performance
# =============================================================================

@st.cache_resource(show_spinner="Loading model and explainer...")
def load_resources():
    """Load model + build SHAP explainer once and cache in memory."""
    model, feature_names = load_model_and_features()
    explainer            = build_explainer(model)
    # Ensure global SHAP plots exist on disk
    regenerate_global_shap_plots(model=model, explainer=explainer)
    return model, feature_names, explainer


model, feature_names, explainer = load_resources()


# =============================================================================
# STEP 4: Sidebar — user inputs for all key employee features
# =============================================================================

with st.sidebar:
    st.markdown("## 👤 Employee Profile")
    st.markdown("Adjust the sliders and dropdowns to match the employee's details.")
    st.markdown("---")

    # ── Personal ──────────────────────────────────────────────────────────────
    st.markdown("### 🧍 Personal")

    age = st.slider(
        "Age", min_value=18, max_value=60, value=35, step=1,
        help="Employee's current age"
    )

    gender = st.selectbox(
        "Gender", options=["Female", "Male"], index=1
    )
    gender_encoded = 1 if gender == "Male" else 0

    marital_status = st.selectbox(
        "Marital Status",
        options=["Divorced", "Married", "Single"], index=1
    )
    marital_encoded = ["Divorced", "Married", "Single"].index(marital_status)

    distance_from_home = st.slider(
        "Distance From Home (km)",
        min_value=1, max_value=29, value=5, step=1,
        help="How far the employee lives from the office"
    )

    st.markdown("---")

    # ── Job Details ───────────────────────────────────────────────────────────
    st.markdown("### 💼 Job Details")

    department = st.selectbox(
        "Department",
        options=["Human Resources", "Research & Development", "Sales"],
        index=1
    )
    dept_encoded = ["Human Resources", "Research & Development", "Sales"].index(department)

    job_role = st.selectbox(
        "Job Role",
        options=[
            "Healthcare Representative", "Human Resources", "Laboratory Technician",
            "Manager", "Manufacturing Director", "Research Director",
            "Research Scientist", "Sales Executive", "Sales Representative"
        ],
        index=6
    )
    job_role_encoded = [
        "Healthcare Representative", "Human Resources", "Laboratory Technician",
        "Manager", "Manufacturing Director", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative"
    ].index(job_role)

    job_level = st.slider(
        "Job Level", min_value=1, max_value=5, value=2, step=1,
        help="1=Entry, 2=Junior, 3=Mid, 4=Senior, 5=Executive"
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        options=["1 — Low", "2 — Medium", "3 — High", "4 — Very High"],
        index=2
    )
    job_sat_encoded = int(job_satisfaction[0])

    job_involvement = st.selectbox(
        "Job Involvement",
        options=["1 — Low", "2 — Medium", "3 — High", "4 — Very High"],
        index=2
    )
    job_inv_encoded = int(job_involvement[0])

    overtime = st.selectbox(
        "OverTime", options=["No", "Yes"], index=0,
        help="Does the employee regularly work overtime?"
    )
    overtime_encoded = 1 if overtime == "Yes" else 0

    business_travel = st.selectbox(
        "Business Travel",
        options=["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        index=1
    )
    travel_encoded = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(business_travel)

    st.markdown("---")

    # ── Compensation ──────────────────────────────────────────────────────────
    st.markdown("### 💰 Compensation")

    monthly_income = st.slider(
        "Monthly Income ($)",
        min_value=1000, max_value=20000, value=5000, step=100,
        help="Employee's gross monthly salary"
    )

    daily_rate = st.slider(
        "Daily Rate", min_value=100, max_value=1500, value=800, step=10
    )

    hourly_rate = st.slider(
        "Hourly Rate", min_value=30, max_value=100, value=65, step=1
    )

    monthly_rate = st.slider(
        "Monthly Rate", min_value=2000, max_value=27000, value=14000, step=100
    )

    percent_salary_hike = st.slider(
        "Percent Salary Hike (%)",
        min_value=11, max_value=25, value=14, step=1
    )

    stock_option_level = st.selectbox(
        "Stock Option Level",
        options=["0 — None", "1 — Low", "2 — Medium", "3 — High"],
        index=1
    )
    stock_encoded = int(stock_option_level[0])

    st.markdown("---")

    # ── Experience & Tenure ───────────────────────────────────────────────────
    st.markdown("### 📅 Experience & Tenure")

    total_working_years = st.slider(
        "Total Working Years",
        min_value=0, max_value=40, value=8, step=1
    )

    years_at_company = st.slider(
        "Years At Company",
        min_value=0, max_value=40, value=5, step=1
    )

    years_in_current_role = st.slider(
        "Years In Current Role",
        min_value=0, max_value=18, value=3, step=1
    )

    years_since_last_promotion = st.slider(
        "Years Since Last Promotion",
        min_value=0, max_value=15, value=1, step=1
    )

    years_with_curr_manager = st.slider(
        "Years With Current Manager",
        min_value=0, max_value=17, value=4, step=1
    )

    num_companies_worked = st.slider(
        "Num Companies Worked",
        min_value=0, max_value=9, value=2, step=1
    )

    training_times_last_year = st.slider(
        "Training Times Last Year",
        min_value=0, max_value=6, value=3, step=1
    )

    st.markdown("---")

    # ── Satisfaction Scores ───────────────────────────────────────────────────
    st.markdown("### 😊 Satisfaction & Balance")

    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        options=["1 — Low", "2 — Medium", "3 — High", "4 — Very High"],
        index=2
    )
    env_sat_encoded = int(environment_satisfaction[0])

    relationship_satisfaction = st.selectbox(
        "Relationship Satisfaction",
        options=["1 — Low", "2 — Medium", "3 — High", "4 — Very High"],
        index=2
    )
    rel_sat_encoded = int(relationship_satisfaction[0])

    work_life_balance = st.selectbox(
        "Work Life Balance",
        options=["1 — Bad", "2 — Good", "3 — Better", "4 — Best"],
        index=2
    )
    wlb_encoded = int(work_life_balance[0])

    st.markdown("---")

    # ── Education ─────────────────────────────────────────────────────────────
    st.markdown("### 🎓 Education")

    education = st.selectbox(
        "Education Level",
        options=[
            "1 — Below College", "2 — College",
            "3 — Bachelor", "4 — Master", "5 — Doctor"
        ],
        index=2
    )
    edu_encoded = int(education[0])

    education_field = st.selectbox(
        "Education Field",
        options=[
            "Human Resources", "Life Sciences", "Marketing",
            "Medical", "Other", "Technical Degree"
        ],
        index=1
    )
    edu_field_encoded = [
        "Human Resources", "Life Sciences", "Marketing",
        "Medical", "Other", "Technical Degree"
    ].index(education_field)

    performance_rating = st.selectbox(
        "Performance Rating",
        options=["1 — Low", "2 — Good", "3 — Excellent", "4 — Outstanding"],
        index=2
    )
    perf_encoded = int(performance_rating[0])


# =============================================================================
# STEP 5: Build input DataFrame in exact feature order the model expects
# =============================================================================

input_data = pd.DataFrame({
    "Age"                      : [age],
    "BusinessTravel"           : [travel_encoded],
    "DailyRate"                : [daily_rate],
    "Department"               : [dept_encoded],
    "DistanceFromHome"         : [distance_from_home],
    "Education"                : [edu_encoded],
    "EducationField"           : [edu_field_encoded],
    "EnvironmentSatisfaction"  : [env_sat_encoded],
    "Gender"                   : [gender_encoded],
    "HourlyRate"               : [hourly_rate],
    "JobInvolvement"           : [job_inv_encoded],
    "JobLevel"                 : [job_level],
    "JobRole"                  : [job_role_encoded],
    "JobSatisfaction"          : [job_sat_encoded],
    "MaritalStatus"            : [marital_encoded],
    "MonthlyIncome"            : [monthly_income],
    "MonthlyRate"              : [monthly_rate],
    "NumCompaniesWorked"       : [num_companies_worked],
    "OverTime"                 : [overtime_encoded],
    "PercentSalaryHike"        : [percent_salary_hike],
    "PerformanceRating"        : [perf_encoded],
    "RelationshipSatisfaction" : [rel_sat_encoded],
    "StockOptionLevel"         : [stock_encoded],
    "TotalWorkingYears"        : [total_working_years],
    "TrainingTimesLastYear"    : [training_times_last_year],
    "WorkLifeBalance"          : [wlb_encoded],
    "YearsAtCompany"           : [years_at_company],
    "YearsInCurrentRole"       : [years_in_current_role],
    "YearsSinceLastPromotion"  : [years_since_last_promotion],
    "YearsWithCurrManager"     : [years_with_curr_manager],
})

# Reorder columns to exactly match model's training feature order
input_data = input_data[feature_names]

# =============================================================================
# STEP 6: Run prediction
# =============================================================================

# predict_proba returns [[prob_class0, prob_class1]]
attrition_prob  = model.predict_proba(input_data)[0][1]   # Probability of leaving
retention_prob  = 1 - attrition_prob                       # Probability of staying
attrition_pct   = attrition_prob * 100
retention_pct   = retention_prob * 100

# Classify risk level with thresholds: Low <35%, Medium 35-60%, High >60%
if attrition_pct < 35:
    risk_level    = "Low"
    badge_class   = "badge-low"
    card_class    = "card-green"
    progress_color = "normal"
    risk_emoji    = "🟢"
elif attrition_pct < 60:
    risk_level    = "Medium"
    badge_class   = "badge-medium"
    card_class    = "card-orange"
    progress_color = "normal"
    risk_emoji    = "🟠"
else:
    risk_level    = "High"
    badge_class   = "badge-high"
    card_class    = "card-red"
    progress_color = "normal"
    risk_emoji    = "🔴"


# =============================================================================
# STEP 7: Main panel — Page header
# =============================================================================

st.markdown("# 🧠 Employee Attrition Risk Predictor")
st.markdown(
    "Powered by **XGBoost + SHAP** · Adjust the sidebar to profile an employee "
    "and instantly see their attrition risk with AI-driven explanations."
)
st.markdown("---")


# =============================================================================
# STEP 8: Three metric cards — Attrition %, Risk Level, Retention %
# =============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    color = "card-red" if risk_level == "High" else ("card-orange" if risk_level == "Medium" else "card-green")
    st.markdown(f"""
    <div class="metric-card {color}">
        <h2>{attrition_pct:.1f}%</h2>
        <p>⚠️ Attrition Risk</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card card-blue">
        <h2 style="font-size:1.5rem;">{risk_emoji} {risk_level}</h2>
        <span class="risk-badge {badge_class}">{risk_level.upper()} RISK</span>
        <p style="margin-top:8px;">Risk Level</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card card-green">
        <h2>{retention_pct:.1f}%</h2>
        <p>✅ Retention Probability</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# STEP 9: Risk progress bar
# =============================================================================

st.markdown('<div class="section-header">📊 Risk Gauge</div>', unsafe_allow_html=True)

# Colour the bar label based on risk
bar_label = f"{risk_emoji} {risk_level} Risk — {attrition_pct:.1f}% probability of attrition"
st.markdown(f"**{bar_label}**")
st.progress(float(attrition_prob))   # st.progress expects a float 0.0–1.0

# Threshold markers as helper text
st.caption("🟢 Low: 0–35%  |  🟠 Medium: 35–60%  |  🔴 High: >60%")
st.markdown("---")


# =============================================================================
# STEP 10: Employee input summary table
# =============================================================================

with st.expander("📋 View Current Employee Input Data", expanded=False):
    display_df = pd.DataFrame({
        "Feature" : feature_names,
        "Value"   : input_data.values[0]
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# =============================================================================
# STEP 11: SHAP Waterfall chart — explains THIS specific employee's prediction
# =============================================================================

st.markdown('<div class="section-header">🔍 Why Is This Employee At Risk? (SHAP Waterfall)</div>', unsafe_allow_html=True)
st.markdown(
    "Each bar shows how much a feature **increases** 🔴 or **decreases** 🔵 "
    "the attrition risk for **this specific employee**."
)

with st.spinner("Generating SHAP explanation..."):
    waterfall_fig = get_shap_waterfall(
        input_data,
        model=model,
        explainer=explainer
    )
    st.pyplot(waterfall_fig, use_container_width=True)
    plt.close()

st.markdown("---")


# =============================================================================
# STEP 12: Global SHAP summary plots
# =============================================================================

st.markdown('<div class="section-header">🌍 Global Feature Importance (All Employees)</div>', unsafe_allow_html=True)
st.markdown("These plots show which features matter most **across the entire dataset**.")

tab1, tab2 = st.tabs(["📊 Feature Importance (Bar)", "🐝 Beeswarm (Direction of Effect)"])

bar_path      = "shap_plots/shap_summary_bar.png"
beeswarm_path = "shap_plots/shap_beeswarm.png"

with tab1:
    if os.path.exists(bar_path):
        st.image(bar_path, caption="Mean |SHAP Value| per feature — higher = more important", use_container_width=True)
    else:
        st.warning("Bar plot not found. Run src/train.py first.")

with tab2:
    if os.path.exists(beeswarm_path):
        st.image(beeswarm_path, caption="Red = pushes toward attrition · Blue = pushes toward retention", use_container_width=True)
    else:
        st.warning("Beeswarm plot not found. Run src/train.py first.")

st.markdown("---")


# =============================================================================
# STEP 13: Footer
# =============================================================================

st.markdown("""
<div style="text-align:center; color:#6c757d; font-size:0.8rem; padding: 12px 0;">
    🧠 Employee Attrition Risk Predictor &nbsp;·&nbsp;
    Built with XGBoost + SHAP + Streamlit &nbsp;·&nbsp;
    Tracked with MLflow on DagsHub
</div>
""", unsafe_allow_html=True)