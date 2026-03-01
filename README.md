# Borrower Risk Intelligence System
> An end-to-end credit risk scoring and explainability framework built from real-world lending experience.

---

## Origin Story

Before building this in Python, I ran a private lending operation for 3 years — manually tracking borrower details, repayment dates, and defaults in Excel. I saw firsthand that lending decisions made purely on intuition led to avoidable defaults. This project is the ML-powered version of that evaluation process: structured, explainable, and data-driven.

---

## Problem Statement

Credit risk models typically output a binary verdict — default or no default. That binary is insufficient for real lending decisions. Every undetected default costs a lender the full principal. Every falsely flagged good borrower costs a relationship. Yet most models optimise for accuracy — which tells you nothing about which error you're making more of, or why.

Real lending decisions require three things: the magnitude of risk, the specific drivers behind it, and a defensible rationale for every approval or rejection.

---

## What This System Does

- Predicts probability of loan default for each borrower
- Converts raw probability into a calibrated risk band (Low / Medium / High / Critical)
- Uses SHAP values to explain the top factors driving each borrower's risk score
- Compares 3 model architectures and selects the best based on recall optimisation
- Serves predictions via an interactive Streamlit dashboard

---

## Dataset

**Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)  
**Size:** 307,511 borrower records, 125 raw features  
**Default rate:** 8.1% (11:1 class imbalance)  
**Class imbalance handling:** `scale_pos_weight` in XGBoost — model trained on real distribution, not resampled

> Dataset is not stored in this repo. See setup instructions below.

---

## Getting the Dataset

**Step 1 — Create a Kaggle account**  
Go to [kaggle.com](https://www.kaggle.com) and sign up if you don't have an account.

**Step 2 — Accept the competition rules**  
Go to [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) and click **Join Competition**. You must accept the rules before the API will allow you to download the data.

**Step 3 — Get your Kaggle API token**  
1. Go to [kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to the **API** section
3. Click **Create New Token**
4. A file called `kaggle.json` will download to your computer

**Step 4 — Download using the notebooks (Google Colab)**  
Open `notebooks/02_feature_engineering.ipynb` in Google Colab. Set `FIRST_TIME = True` at the top of the first cell. When you run it, it will prompt you to upload your `kaggle.json`. The dataset downloads automatically, saves to your Google Drive, and all subsequent notebooks load from Drive.

After the first successful run, set `FIRST_TIME = False` — the dataset will load from Drive in seconds every future session.

**Dataset saved to Drive at:**
```
/content/drive/MyDrive/borrower-risk-intelligence/data/application_train.csv
```

---

## Feature Engineering

Features designed from lending intuition, not statistical iteration:

| Feature | Logic |
|---|---|
| `CREDIT_INCOME_RATIO` | Loan amount vs annual income — primary repayment stress signal |
| `ANNUITY_INCOME_RATIO` | Monthly repayment as fraction of income |
| `CREDIT_GOODS_RATIO` | Fraction of goods price being financed — borrowers at 100% have no skin in the game |
| `ANNUITY_CREDIT_RATIO` | Effective repayment period signal |
| `EMPLOYMENT_YEARS` | Shorter tenure = higher risk |
| `EXT_SOURCE_MEAN` | Aggregate of external credit bureau scores |
| `INCOME_PER_PERSON` | Income adjusted for family size — shared financial burden |

41 features with >50% missing values were dropped. 13 categorical columns label-encoded. Final feature set: 95 features.

---

## Models Compared

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.7470 | 0.1596 | 0.6727 | 0.2580 |
| Random Forest | 0.7472 | 0.1776 | 0.5972 | 0.2738 |
| XGBoost ✅ | 0.7644 | 0.1791 | 0.6568 | 0.2815 |
| XGBoost (tuned, t=0.66) | 0.7644 | 0.2550 | 0.4121 | 0.3150 |

**Cross-validation:** 5-fold stratified CV on 150k sample — ROC-AUC 0.7501 ± 0.0039

> CV was run on a 150,000-row stratified sample of the training set. A sample of this size produces AUC estimates within ~0.002 of full-dataset CV while keeping runtime manageable. The goal is variance estimation — confirming the model generalises — not squeezing out an extra decimal place.

**Why recall was the primary optimisation target:**

In credit risk, the two error types have different costs:
- **False Negative** (missing a defaulter) → lose the full loan principal
- **False Positive** (flagging a good borrower) → lose one customer relationship

The model threshold was tuned from 0.5 to 0.66 to find the best F1 on the default class. XGBoost was selected for highest ROC-AUC and best precision-recall balance across all three models.

---

## SHAP Explainability

Top 10 features by mean absolute SHAP value:

| Feature | Mean \|SHAP\| |
|---|---|
| EXT_SOURCE_MEAN | 0.4443 |
| ANNUITY_CREDIT_RATIO | 0.1723 |
| CODE_GENDER | 0.1528 |
| CREDIT_GOODS_RATIO | 0.1157 |
| NAME_EDUCATION_TYPE | 0.1135 |
| EXT_SOURCE_3 | 0.1013 |
| EXT_SOURCE_PRODUCT | 0.0962 |
| FLAG_OWN_CAR | 0.0885 |
| AMT_GOODS_PRICE | 0.0841 |
| AMT_ANNUITY | 0.0691 |

External credit scores dominate — expected for any credit risk model. Domain-engineered features (`ANNUITY_CREDIT_RATIO`, `CREDIT_GOODS_RATIO`) appear in the top 10, confirming that lending intuition translated into genuine model signal.

Every prediction comes with a SHAP waterfall chart showing which features pushed the risk score up or down — giving a lender a defensible reason for every decision.

---

## Risk Scoring Framework

| Probability | Risk Band | Recommended Action |
|---|---|---|
| < 0.20 | 🟢 Low | Approve |
| 0.20 – 0.40 | 🟡 Medium | Review |
| 0.40 – 0.60 | 🔴 High | Decline or require collateral |
| > 0.60 | ⛔ Critical | Decline |

**Test set distribution:**

| Band | Borrowers | % |
|---|---|---|
| Low | 13,259 | 21.6% |
| Medium | 21,813 | 35.5% |
| High | 15,026 | 24.4% |
| Critical | 11,405 | 18.5% |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Modelling | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Deployment | Hugging Face Spaces |

---

## Project Structure

```
borrower-risk-intelligence/
│
├── data/
│   └── README.md                    # Instructions to download from Kaggle
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # Feature engineering + train/test split
│   ├── 03_model_comparison.ipynb    # Model training and comparison
│   └── 04_shap_explainability.ipynb # SHAP analysis and risk scoring
│
├── app/
│   └── streamlit_app.py             # Interactive risk scoring dashboard
│
├── models/
│   └── xgboost_final.pkl            # Saved model
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/Sachin-kalaiselvan/borrower-risk-intelligence
cd borrower-risk-intelligence
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Or try the **live demo** → [Hugging Face Spaces link coming soon]

---

## Key Learnings

Running a real lending operation taught me that default risk is rarely about a single factor — it is about the combination of behavioural and financial signals. This project formalises that intuition into a reproducible, explainable ML system.

---

**Sachin JK** — [LinkedIn](https://www.linkedin.com/in/sachinjk11/) | [GitHub](https://github.com/Sachin-kalaiselvan)
