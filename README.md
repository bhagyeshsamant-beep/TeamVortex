# Financial Fraud Detection System

A machine learning system that analyzes mobile money transactions and
predicts whether they are fraudulent, using the PaySim simulated dataset.

## Problem

Digital payment systems process millions of transactions daily, making
manual fraud review impossible. This project builds an ML pipeline that
flags suspicious transactions automatically, with a human-readable
explanation for every flag.

## Key Findings from Data Exploration

- The dataset is **extremely imbalanced**: only 0.13% of transactions are
  fraudulent (8,213 out of 6.36M).
- Fraud **only occurs** in `TRANSFER` and `CASH_OUT` transaction types —
  never in `PAYMENT`, `CASH_IN`, or `DEBIT`. This matches real-world logic:
  fraud requires moving money *out* of an account.
- Based on this, we filter the dataset to TRANSFER/CASH_OUT only before
  training, which removes noise and doubles the effective fraud rate the
  model has to learn from (0.13% → 0.30%).

## Approach

1. **EDA** (`src/01_eda.py`) — understand class imbalance and transaction
   patterns before touching any model.
2. **Preprocessing & feature engineering** (`src/02_preprocess.py`) —
   engineer balance-mismatch features (`errorBalanceOrig`,
   `errorBalanceDest`) that directly capture inconsistencies indicative of
   fraud, rather than relying on the model to discover them from raw
   balances alone.
3. **Modeling** (`src/03_train_model.py`) — train and compare:
   - Logistic Regression (baseline, `class_weight="balanced"`)
   - HistGradientBoostingClassifier (main model, `class_weight="balanced"`)
4. **Explainability** (`src/04_explain.py`) — global permutation feature
   importance, plus a per-transaction explanation comparing a flagged
   transaction's values against typical legitimate transactions.
5. **LLM explanation layer** (`src/05_llm_explain.py`) — a local LLM
   (Bonsai 27B, via Ollama) rephrases the statistical factors above into
   one plain-English sentence for a non-technical reviewer. The
   classifier makes the decision; the LLM only explains it in words —
   this keeps the ML decision-making auditable while making the output
   more usable. Falls back to the statistical explanation automatically
   if the LLM isn't reachable, so a live demo never breaks.
6. **UI** (`app.py`) — a Streamlit app for live, interactive predictions,
   with an optional "Generate AI explanation" toggle for the LLM layer.

### Setting up the local LLM (optional)

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull the model (5.9GB, fits fine in 16GB RAM)
ollama pull hf.co/prism-ml/Ternary-Bonsai-27B-gguf
# 3. Make sure Ollama is running, then test:
python3 src/05_llm_explain.py
```

If inference feels slow on your hardware, swap to a much lighter model —
edit `MODEL_NAME` in `src/05_llm_explain.py` and `LLM_MODEL_NAME` in
`app.py`:

```bash
ollama pull phi4-mini
```

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.092 | 0.954 | 0.168 | 0.993 |
| HistGradientBoosting (main model) | 0.986 | 0.997 | 0.991 | 0.999 |

The baseline catches most fraud but at the cost of huge false-positive
volume (unusable in practice — it would flag ~15k legitimate transactions
per 550k). The upgraded model achieves both high recall and high precision,
making it practical for real deployment.

**Top predictive feature:** `errorBalanceOrig` (our engineered feature)
ranks #1 in permutation importance — validating the feature engineering
approach over relying on raw columns alone.

## Project Structure

```
├── data/
│   └── paysim_dataset.csv       # raw dataset (not committed if large — see .gitignore)
├── src/
│   ├── 01_eda.py                # exploratory data analysis
│   ├── 02_preprocess.py         # cleaning + feature engineering
│   ├── 03_train_model.py        # train baseline + main model
│   ├── 04_explain.py            # feature importance + per-transaction explanations
│   └── 05_llm_explain.py        # local LLM (Bonsai 27B) natural-language explanations
├── models/                      # saved trained models (generated)
├── outputs/                     # generated reports/charts
├── app.py                       # Streamlit UI
├── requirements.txt
└── README.md
```

## How to Run

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

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Logistic Regression, HistGradientBoostingClassifier,
  permutation importance)
- Streamlit (UI)


## Future Improvements

- Real-time streaming inference (Kafka + model serving)
- SHAP-based explanations (swap in `shap.TreeExplainer` given internet access)
- Graph-based features (account network analysis for mule-account detection)
- Threshold tuning per business cost of false positives vs false negatives
