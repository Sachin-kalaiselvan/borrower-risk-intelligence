import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

st.set_page_config(
    page_title="Borrower Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0a0e1a; color: #e8eaf0; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #e8eaf0; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model     = pickle.load(open(os.path.join(BASE_DIR, "models", "xgboost_final.pkl"), 'rb'))
    threshold = pickle.load(open(os.path.join(BASE_DIR, "models", "best_threshold.pkl"), 'rb'))
    features  = pickle.load(open(os.path.join(BASE_DIR, "models", "feature_list.pkl"), 'rb'))
    explainer = shap.TreeExplainer(model)
    return model, threshold, features, explainer

try:
    model, best_threshold, feature_list, explainer = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

def get_risk_band(prob):
    if prob < 0.20:   return 'LOW',      '#22c55e', 'Approve'
    elif prob < 0.40: return 'MEDIUM',   '#f59e0b', 'Review application'
    elif prob < 0.60: return 'HIGH',     '#f97316', 'Decline or require collateral'
    else:             return 'CRITICAL', '#ef4444', 'Decline'

def build_feature_vector(inputs, feature_list):
    row = {f: 0.0 for f in feature_list}
    row['AMT_INCOME_TOTAL']    = inputs['annual_income']
    row['AMT_CREDIT']          = inputs['loan_amount']
    row['AMT_ANNUITY']         = inputs['monthly_annuity']
    row['AMT_GOODS_PRICE']     = inputs['goods_price']
    row['DAYS_BIRTH']          = -inputs['age'] * 365
    row['DAYS_EMPLOYED']       = -inputs['employment_years'] * 365
    row['CNT_FAM_MEMBERS']     = inputs['family_members']
    row['EXT_SOURCE_2']        = inputs['ext_score_2']
    row['EXT_SOURCE_3']        = inputs['ext_score_3']
    row['CODE_GENDER']         = 0 if inputs['gender'] == 'Male' else 1
    row['FLAG_OWN_CAR']        = 1 if inputs['owns_car'] else 0
    row['FLAG_OWN_REALTY']     = 1 if inputs['owns_realty'] else 0
    row['CNT_CHILDREN']        = inputs['children']
    row['AGE_YEARS']           = inputs['age']
    row['EMPLOYMENT_YEARS']    = inputs['employment_years']
    row['CREDIT_INCOME_RATIO'] = inputs['loan_amount'] / max(inputs['annual_income'], 1)
    row['ANNUITY_INCOME_RATIO']= inputs['monthly_annuity'] / max(inputs['annual_income'], 1)
    row['CREDIT_GOODS_RATIO']  = inputs['loan_amount'] / max(inputs['goods_price'], 1)
    row['ANNUITY_CREDIT_RATIO']= inputs['monthly_annuity'] / max(inputs['loan_amount'], 1)
    row['INCOME_PER_PERSON']   = inputs['annual_income'] / max(inputs['family_members'], 1)
    ext_vals = [inputs['ext_score_2'], inputs['ext_score_3']]
    row['EXT_SOURCE_MEAN']     = np.mean(ext_vals)
    row['EXT_SOURCE_MIN']      = np.min(ext_vals)
    row['EXT_SOURCE_PRODUCT']  = np.prod(ext_vals)
    return pd.DataFrame([row])[feature_list].fillna(0)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Borrower Risk Intelligence")
st.markdown("Credit risk scoring with SHAP explainability — built from real lending experience.")
st.divider()

if not model_loaded:
    st.error("Model files not found.")
    st.code(load_error)
    st.stop()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Borrower Details")

inputs = {
    "annual_income":    st.sidebar.number_input("Annual Income (₹)", min_value=10000, max_value=10000000, value=300000, step=10000),
    "loan_amount":      st.sidebar.number_input("Loan Amount (₹)", min_value=10000, max_value=5000000, value=500000, step=10000),
    "monthly_annuity":  st.sidebar.number_input("Monthly Repayment (₹)", min_value=1000, max_value=200000, value=15000, step=1000),
    "goods_price":      st.sidebar.number_input("Goods Price (₹)", min_value=10000, max_value=5000000, value=450000, step=10000),
    "age":              st.sidebar.slider("Age", 18, 70, 35),
    "employment_years": st.sidebar.slider("Years Employed", 0, 40, 5),
    "family_members":   st.sidebar.slider("Family Members", 1, 10, 3),
    "children":         st.sidebar.slider("Number of Children", 0, 5, 1),
    "ext_score_2":      st.sidebar.slider("External Credit Score 2", 0.0, 1.0, 0.6),
    "ext_score_3":      st.sidebar.slider("External Credit Score 3", 0.0, 1.0, 0.5),
    "gender":           st.sidebar.selectbox("Gender", ["Male", "Female"]),
    "owns_car":         st.sidebar.checkbox("Owns a Car", value=False),
    "owns_realty":      st.sidebar.checkbox("Owns Property", value=True),
}

# ── Predict ───────────────────────────────────────────────────────────────────
if st.sidebar.button("Assess Risk", use_container_width=True):
    df = build_feature_vector(inputs, feature_list)
    prob      = model.predict_proba(df)[0, 1]
    band, color, action = get_risk_band(prob)

    # ── Results ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{prob:.1%}")
    col2.metric("Risk Band", band)
    col3.metric("Recommended Action", action)

    st.divider()

    # ── SHAP waterfall chart ──────────────────────────────────────────────────
    st.subheader("What drove this score?")
    shap_values = explainer.shap_values(df)
    vals   = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    top_idx = np.argsort(np.abs(vals))[-10:][::-1]
    top_features = [feature_list[i] for i in top_idx]
    top_vals     = [vals[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#ef4444' if v > 0 else '#22c55e' for v in top_vals]
    ax.barh(top_features[::-1], top_vals[::-1], color=colors[::-1])
    ax.axvline(0, color='white', linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on default probability)", color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0a0e1a')
    fig.patch.set_facecolor('#0a0e1a')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    st.pyplot(fig)
    plt.close()

    st.caption("Red bars increase default risk. Green bars reduce it.")
else:
    st.info("Fill in the borrower details in the sidebar and click **Assess Risk**.")
