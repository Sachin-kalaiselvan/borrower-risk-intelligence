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

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #0a0e1a;
    color: #e8eaf0;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    color: #e8eaf0;
}

.metric-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

.risk-LOW {
    background: #0d2618;
    border-left: 4px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 20px 24px;
}

.risk-MEDIUM {
    background: #1f1a0a;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 20px 24px;
}

.risk-HIGH {
    background: #1f0e0a;
    border-left: 4px solid #f97316;
    border-radius: 0 8px 8px 0;
    padding: 20px 24px;
}

.risk-CRITICAL {
    background: #1f0a0a;
    border-left: 4px solid #ef4444;
    border-radius: 0 8px 8px 0;
    padding: 20px 24px;
}

.band-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: 0.1em;
}

.driver-row {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 10px 16px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    display: flex;
    justify-content: space-between;
}

.increases { color: #ef4444; }
.decreases { color: #22c55e; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4b6080;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2a3a;
}

div[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #1e2a3a;
}

.stSlider > div > div > div > div {
    background-color: #3b82f6 !important;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path     = 'models/xgboost_final.pkl'
    threshold_path = 'models/best_threshold.pkl'
    features_path  = 'models/feature_list.pkl'

    with open(model_path,     'rb') as f: model          = pickle.load(f)
    with open(threshold_path, 'rb') as f: best_threshold = pickle.load(f)
    with open(features_path,  'rb') as f: feature_list   = pickle.load(f)

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

    # Direct mappings
    row['AMT_INCOME_TOTAL']      = inputs['annual_income']
    row['AMT_CREDIT']            = inputs['loan_amount']
    row['AMT_ANNUITY']           = inputs['monthly_annuity']
    row['AMT_GOODS_PRICE']       = inputs['goods_price']
    row['DAYS_BIRTH']            = -inputs['age'] * 365
    row['DAYS_EMPLOYED']         = -inputs['employment_years'] * 365
    row['CNT_FAM_MEMBERS']       = inputs['family_members']
    row['EXT_SOURCE_2']          = inputs['ext_score_2']
    row['EXT_SOURCE_3']          = inputs['ext_score_3']
    row['CODE_GENDER']           = 0 if inputs['gender'] == 'Male' else 1
    row['FLAG_OWN_CAR']          = 1 if inputs['owns_car'] else 0
    row['FLAG_OWN_REALTY']       = 1 if inputs['owns_realty'] else 0
    row['CNT_CHILDREN']          = inputs['children']

    # Engineered features
    row['AGE_YEARS']             = inputs['age']
    row['EMPLOYMENT_YEARS']      = inputs['employment_years']
    row['CREDIT_INCOME_RATIO']   = inputs['loan_amount'] / max(inputs['annual_income'], 1)
    row['ANNUITY_INCOME_RATIO']  = inputs['monthly_annuity'] / max(inputs['annual_income'], 1)
    row['CREDIT_GOODS_RATIO']    = inputs['loan_amount'] / max(inputs['goods_price'], 1)
    row['ANNUITY_CREDIT_RATIO']  = inputs['monthly_annuity'] / max(inputs['loan_amount'], 1)
    row['INCOME_PER_PERSON']     = inputs['annual_income'] / max(inputs['family_members'], 1)

    ext_vals = [inputs['ext_score_2'], inputs['ext_score_3']]
    row['EXT_SOURCE_MEAN']       = np.mean(ext_vals)
    row['EXT_SOURCE_MIN']        = np.min(ext_vals)
    row['EXT_SOURCE_PRODUCT']    = np.prod(ext_vals)

    row['AGE_BUCKET'] = (
        0 if inputs['age'] < 25 else
        1 if inputs['age'] < 35 else
        2 if inputs['age'] < 45 else
        3 if inputs['age'] < 55 else 4
    )

    row['DOCUMENT_COUNT']             = inputs['doc_count']
    row['EMPLOYMENT_YEARS_MISSING']   = 0
    row['EXT_SOURCE_3_MISSING']       = 0
    row['AMT_GOODS_PRICE_MISSING']    = 0
    row['AMT_ANNUITY_MISSING']        = 0

    return pd.DataFrame([row])[feature_list].fillna(0)


# ── Sidebar — borrower inputs ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Borrower Profile</div>', unsafe_allow_html=True)

    gender           = st.selectbox('Gender',           ['Male', 'Female'])
    age              = st.slider('Age',                  18, 70, 35)
    children         = st.slider('Number of Children',  0, 10, 0)
    family_members   = st.slider('Family Members',       1, 10, 2)
    owns_car         = st.checkbox('Owns Car',           value=False)
    owns_realty      = st.checkbox('Owns Realty',        value=True)

    st.markdown('<div class="section-header">Financial Details</div>', unsafe_allow_html=True)

    annual_income    = st.number_input('Annual Income (₹)',     min_value=50000,    max_value=10000000, value=300000,  step=10000)
    loan_amount      = st.number_input('Loan Amount (₹)',       min_value=10000,    max_value=5000000,  value=500000,  step=10000)
    monthly_annuity  = st.number_input('Monthly Repayment (₹)', min_value=1000,     max_value=200000,   value=15000,   step=500)
    goods_price      = st.number_input('Goods Price (₹)',       min_value=10000,    max_value=5000000,  value=450000,  step=10000)

    st.markdown('<div class="section-header">Employment & Credit</div>', unsafe_allow_html=True)

    employment_years = st.slider('Employment Duration (Years)', 0.0, 40.0, 3.0, 0.5)
    ext_score_2      = st.slider('External Credit Score 2',     0.0, 1.0,  0.5, 0.01)
    ext_score_3      = st.slider('External Credit Score 3',     0.0, 1.0,  0.5, 0.01)
    doc_count        = st.slider('Documents Submitted',         0,   20,   3)

    score_btn = st.button('Score This Borrower', use_container_width=True)


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# Borrower Risk Intelligence")
st.markdown("Credit risk scoring with SHAP explainability — built from real lending experience.")
st.markdown("---")

if not model_loaded:
    st.error(f"Model files not found. Ensure `models/` folder contains `xgboost_final.pkl`, `best_threshold.pkl`, and `feature_list.pkl`.")
    st.info("Run the notebooks first to generate the model files, then copy them into the `models/` folder.")
    st.stop()

# Run scoring on button press or on first load with defaults
inputs = {
    'gender': gender, 'age': age, 'children': children,
    'family_members': family_members, 'owns_car': owns_car,
    'owns_realty': owns_realty, 'annual_income': annual_income,
    'loan_amount': loan_amount, 'monthly_annuity': monthly_annuity,
    'goods_price': goods_price, 'employment_years': employment_years,
    'ext_score_2': ext_score_2, 'ext_score_3': ext_score_3,
    'doc_count': doc_count
}

X_input     = build_feature_vector(inputs, feature_list)
prob        = model.predict_proba(X_input)[0, 1]
band, color, action = get_risk_band(prob)
shap_vals   = explainer.shap_values(X_input)[0]

# ── Risk result ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"""
    <div class="risk-{band}">
        <div class="section-header">Risk Assessment</div>
        <div class="band-label" style="color:{color}">{band}</div>
        <div style="font-size:2.5rem; font-family:'IBM Plex Mono',monospace; margin: 8px 0;">
            {prob:.1%}
        </div>
        <div style="color:#8899aa; font-size:0.9rem;">probability of default</div>
        <div style="margin-top:16px; padding-top:16px; border-top:1px solid #1e2a3a;">
            <div style="color:#8899aa; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;">Recommendation</div>
            <div style="font-weight:600; margin-top:4px;">{action}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key ratios
    st.markdown('<div class="section-header" style="margin-top:20px;">Key Ratios</div>', unsafe_allow_html=True)
    credit_income = loan_amount / max(annual_income, 1)
    annuity_income = monthly_annuity / max(annual_income, 1)
    credit_goods = loan_amount / max(goods_price, 1)

    for label, val, threshold, higher_is_worse in [
        ('Credit / Income',   credit_income,  3.0,  True),
        ('Annuity / Income',  annuity_income, 0.35, True),
        ('Credit / Goods',    credit_goods,   1.0,  True),
    ]:
        flag = '🔴' if (val > threshold) == higher_is_worse else '🟢'
        st.markdown(f"""
        <div class="metric-card" style="padding:12px 16px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.85rem; color:#8899aa;">{label}</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-weight:600;">{flag} {val:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">Top Risk Drivers</div>', unsafe_allow_html=True)

    shap_series = pd.Series(shap_vals, index=feature_list)
    top_drivers = shap_series.abs().sort_values(ascending=False).head(8)

    max_shap = top_drivers.values[0]
    for feat in top_drivers.index:
        val      = shap_series[feat]
        bar_pct  = abs(val) / max_shap * 100
        direction = 'increases' if val > 0 else 'decreases'
        dir_color = '#ef4444' if val > 0 else '#22c55e'
        arrow     = '↑' if val > 0 else '↓'

        st.markdown(f"""
        <div style="margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem;">{feat}</span>
                <span style="font-size:0.8rem; color:{dir_color}; font-weight:600;">{arrow} {direction} risk</span>
            </div>
            <div style="background:#1e2a3a; border-radius:4px; height:6px;">
                <div style="background:{dir_color}; width:{bar_pct:.0f}%; height:6px; border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # SHAP waterfall
    st.markdown('<div class="section-header" style="margin-top:24px;">SHAP Waterfall</div>', unsafe_allow_html=True)

    explanation = shap.Explanation(
        values        = shap_vals,
        base_values   = explainer.expected_value,
        data          = X_input.values[0],
        feature_names = feature_list
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    shap.waterfall_plot(explanation, max_display=10, show=False)

    plt.gcf().patch.set_facecolor('#111827')
    for ax_ in plt.gcf().axes:
        ax_.set_facecolor('#111827')
        ax_.tick_params(colors='#8899aa')
        ax_.xaxis.label.set_color('#8899aa')
        for spine in ax_.spines.values():
            spine.set_color('#1e2a3a')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Model info footer ─────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div style="color:#4b6080;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;">Model</div><div style="font-family:\'IBM Plex Mono\',monospace;font-weight:600;margin-top:4px;">XGBoost</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div style="color:#4b6080;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;">ROC-AUC</div><div style="font-family:\'IBM Plex Mono\',monospace;font-weight:600;margin-top:4px;">0.7644</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div style="color:#4b6080;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;">CV Score</div><div style="font-family:\'IBM Plex Mono\',monospace;font-weight:600;margin-top:4px;">0.7501 ± 0.004</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div style="color:#4b6080;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;">Threshold</div><div style="font-family:\'IBM Plex Mono\',monospace;font-weight:600;margin-top:4px;">{:.2f}</div></div>'.format(best_threshold), unsafe_allow_html=True)
