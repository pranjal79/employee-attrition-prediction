# 🧠 Employee Attrition Risk Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![DagsHub](https://img.shields.io/badge/DagsHub-Tracked-orange)](https://dagshub.com)
[![MLflow](https://img.shields.io/badge/MLflow-Experiments-blue)](https://mlflow.org)

Predicts whether an employee is likely to leave the company using IBM HR Analytics data.

## 🚀 Tech Stack
- **ML**: scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Explainability**: SHAP
- **Experiment Tracking**: MLflow + DagsHub
- **Data Versioning**: DVC + DagsHub
- **Deployment**: Streamlit Cloud

## 📁 Project Structure
\`\`\`
employee-attrition-prediction/
├── data/           # Raw & processed data (DVC tracked)
├── src/            # Preprocessing, training, SHAP scripts
├── models/         # Saved model artifacts
├── shap_plots/     # SHAP visualization outputs
├── app.py          # Streamlit web app
└── requirements.txt
\`\`\`

## ⚙️ Setup
\`\`\`bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
streamlit run app.py
\`\`\`

## 📊 Models Trained
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost (Best — tuned with GridSearchCV)

## 🌐 Live App
[https://employee-attrition-predictionproject.streamlit.app/](#) ← update after deployment