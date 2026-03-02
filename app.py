import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Borrower Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0a0e1a; color: #e8eaf0; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #e8eaf0; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load model (FIXED PATH VERSION) ──────────────────────────────────────────
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path     = os.path.join(BASE_DIR, "models", "xgboost_final.pkl")
    threshold_path = os.path.join(BASE_DIR, "models", "best_threshold.pkl")
    features_path  = os.path.join(BASE_DIR, "models", "feature_list.pkl")

    model          = pickle.load(open(model_path, 'rb'))
    best_threshold = pickle.load(open(threshold_path, 'rb'))
    feature_list   = pickle.load(open(features_path, 'rb'))

    explainer = shap.TreeExplainer(model)

    return model, best_threshold, feature_list, explainer


try:
    model, best_threshold, feature_list, explainer = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ── Risk band helper ──────────────────────────────────────────────────────────
def get_risk_band(prob):
    if prob < 0.20:   return 'LOW',      '#22c55e', 'Approve'
    elif prob < 0.40: return 'MEDIUM',   '#f59e0b', 'Review application'
    elif prob < 0.60: return 'HIGH',     '#f97316', 'Decline or require collateral'
    else:             return 'CRITICAL', '#ef4444', 'Decline'


# ── Build feature vector ──────────────────────────────────────────────────────
def build_feature_vector(inputs, feature_list):
    row = {f: 0.0 for f in feature_list}

    row['AMT_INCOME_TOTAL'] = inputs['annual_income']
    row['AMT_CREDIT'] = inputs['loan_amount']
    row['AMT_ANNUITY'] = inputs['monthly_annuity']
    row['AMT_GOODS_PRICE'] = inputs['goods_price']
    row['DAYS_BIRTH'] = -inputs['age'] * 365
    row['DAYS_EMPLOYED'] = -inputs['employment_years'] * 365
    row['CNT_FAM_MEMBERS'] = inputs['family_members']
    row['EXT_SOURCE_2'] = inputs['ext_score_2']
    row['EXT_SOURCE_3'] = inputs['ext_score_3']
    row['CODE_GENDER'] = 0 if inputs['gender'] == 'Male' else 1
    row['FLAG_OWN_CAR'] = 1 if inputs['owns_car'] else 0
    row['FLAG_OWN_REALTY'] = 1 if inputs['owns_realty'] else 0
    row['CNT_CHILDREN'] = inputs['children']

    row['AGE_YEARS'] = inputs['age']
    row['EMPLOYMENT_YEARS'] = inputs['employment_years']
    row['CREDIT_INCOME_RATIO'] = inputs['loan_amount'] / max(inputs['annual_income'], 1)
    row['ANNUITY_INCOME_RATIO'] = inputs['monthly_annuity'] / max(inputs['annual_income'], 1)
    row['CREDIT_GOODS_RATIO'] = inputs['loan_amount'] / max(inputs['goods_price'], 1)
    row['ANNUITY_CREDIT_RATIO'] = inputs['monthly_annuity'] / max(inputs['loan_amount'], 1)
    row['INCOME_PER_PERSON'] = inputs['annual_income'] / max(inputs['family_members'], 1)

    ext_vals = [inputs['ext_score_2'], inputs['ext_score_3']]
    row['EXT_SOURCE_MEAN'] = np.mean(ext_vals)
    row['EXT_SOURCE_MIN'] = np.min(ext_vals)
    row['EXT_SOURCE_PRODUCT'] = np.prod(ext_vals)

    return pd.DataFrame([row])[feature_list].fillna(0)


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("Borrower Risk Intelligence")
st.markdown("Credit risk scoring with SHAP explainability — built from real lending experience.")

if not model_loaded:
    st.error("Model files not found or failed to load.")
    st.code(load_error)
    st.stop()

# Simple test scoring to verify model loads
sample_input = pd.DataFrame([dict.fromkeys(feature_list, 0)])
prob = model.predict_proba(sample_input)[0, 1]
st.success("Model loaded successfully 🎉")
st.write("Test Probability:", prob)

st.markdown("Your full UI continues below…")
