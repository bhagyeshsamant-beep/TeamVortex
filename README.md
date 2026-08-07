# Financial Fraud Detection System

Digital payment platforms process millions of transactions a day, and manually reviewing all of them for fraud isn't realistic. This project trains a model on PaySim's simulated mobile money data to flag fraudulent transactions automatically — and, just as important, explain *why* it flagged them, since a fraud model nobody trusts doesn't get used.

## What we found in the data

The first surprise: fraud is rare. Only 8,213 of 6.36M transactions (0.13%) are labeled fraudulent, so accuracy is a useless metric here — a model that never predicts fraud would still be "99.87% accurate."

The second, more useful finding: fraud only ever shows up in `TRANSFER` and `CASH_OUT` transactions, never in `PAYMENT`, `CASH_IN`, or `DEBIT`. That lines up with how fraud actually works — money has to leave an account for it to be a problem. So we filtered the dataset down to just those two types before training. It cuts out a lot of noise and roughly doubles the effective fraud rate the model has to learn from, from 0.13% to 0.30%.

## Approach

1. **EDA** (`src/01_eda.py`) — dig into class imbalance and transaction patterns before touching a model.
2. **Preprocessing & feature engineering** (`src/02_preprocess.py`) — engineer balance-mismatch features (`errorBalanceOrig`, `errorBalanceDest`) that directly capture the kind of inconsistency that signals fraud, instead of hoping the model finds that pattern on its own from raw balances.
3. **Modeling** (`src/03_train_model.py`) — train and compare two models:
   - Logistic Regression, as a simple baseline (`class_weight="balanced"`)
   - HistGradientBoostingClassifier, the main model (`class_weight="balanced"`)
4. **Explainability** (`src/04_explain.py`) — global permutation feature importance, plus a per-transaction explanation that compares a flagged transaction against what a typical legitimate one looks like.
5. **UI** (`app.py`) — a Streamlit app for live predictions.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.092 | 0.954 | 0.168 | 0.993 |
| HistGradientBoosting (main model) | 0.986 | 0.997 | 0.991 | 0.999 |

The baseline catches almost all the fraud, but at a real cost: it would flag around 15,000 legitimate transactions per 550,000, which is a lot of false alarms for a human team to sort through. The gradient-boosted model gets both precision and recall above 0.98 — the difference between "technically works" and "could actually run in production."

`errorBalanceOrig`, our engineered feature, comes out as the single most important feature by permutation importance — a decent sign the feature engineering step mattered more than just throwing raw columns at a stronger model.

## Project structure

```
├── data/
│   └── paysim_dataset.csv       # raw dataset — gitignored, not included in this repo
├── src/
│   ├── 01_eda.py                # exploratory data analysis
│   ├── 02_preprocess.py         # cleaning + feature engineering
│   ├── 03_train_model.py        # train baseline + main model
│   └── 04_explain.py            # feature importance + per-transaction explanations
├── models/                      # saved trained models (generated)
├── outputs/                     # generated reports/charts
├── app.py                       # Streamlit UI
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the PaySim dataset at data/paysim_dataset.csv

# 3. Run the pipeline in order
python3 src/01_eda.py
python3 src/02_preprocess.py
python3 src/03_train_model.py
python3 src/04_explain.py

# 4. Launch the interactive app
streamlit run app.py
```

## Tech stack

Python, pandas, NumPy, scikit-learn (Logistic Regression, HistGradientBoostingClassifier, permutation importance), Streamlit.

## What we'd add next

- Real-time streaming inference (Kafka + model serving)
- SHAP-based explanations, swapping in `shap.TreeExplainer`
- Graph-based features — account network analysis for catching mule accounts
- Threshold tuning based on the actual business cost of false positives vs. false negatives
