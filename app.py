"""
app.py — Streamlit UI for the Fraud Detection system.

Run locally with:
    streamlit run app.py

Lets a user enter transaction details manually and get:
  - Fraud / Legitimate prediction
  - Confidence score
  - Plain-English explanation of the top contributing factors

This mirrors 04_explain.py's per-transaction z-score explanation approach,
but wired up for live, single-transaction input instead of batch test data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import json
import re

def _clean_llm_text(text):
    """Make arbitrary LLM output safe to render as Streamlit markdown.
    Small/quantized models occasionally substitute underscores for spaces
    or sprinkle in stray markdown characters -- left alone, Streamlit's
    markdown renderer treats those as formatting syntax (e.g. underscores
    as italics), swallowing them and gluing words together. This restores
    real spacing and escapes anything markdown could still misinterpret."""
    text = re.sub(r"(?<=\w)_(?=\w)", " ", text)  # underscore-as-space glitch
    for ch in ("*", "`", "#", "[", "]"):
        text = text.replace(ch, "")
    return text

st.set_page_config(page_title="Fraud Detection System", page_icon="🔍", layout="centered")

# --- Local LLM explanation layer (optional, via Ollama) ---
# See src/05_llm_explain.py for the full explanation of why this exists
# and how it's used. Swap MODEL_NAME to "phi4-mini" if Bonsai is too slow
# on your hardware.
LLM_MODEL_NAME = "phi4-mini"
OLLAMA_URL = "http://localhost:11434/api/chat"

# If a transaction's most extreme feature is this many standard deviations
# from a typical legit transaction, treat it as outside anything the model
# was trained on -- the prediction is likely meaningless at that point,
# since tree-based models don't extrapolate past what they've seen.
OOD_THRESHOLD_STD = 10

FEATURE_DESCRIPTIONS = {
    "amount": "the size of the transaction",
    "oldbalanceOrg": "the sender's balance before the transaction",
    "newbalanceOrig": "the sender's balance after the transaction",
    "oldbalanceDest": "the receiver's balance before the transaction",
    "newbalanceDest": "the receiver's balance after the transaction",
    "step": "when the transaction happened, in the simulation's time steps",
    "type_TRANSFER": "this being a transfer rather than a cash-out",
    "errorBalanceOrig": "a mismatch between what left the sender's account and the transaction amount",
    "errorBalanceDest": "a mismatch between what arrived in the receiver's account and the transaction amount",
    "origAccountEmptied": "the sender's account being fully drained to zero",
    "destBalanceZeroBoth": "the receiver's account being empty both before and after",
}

def llm_explain(prediction, confidence, factors, is_ood=False, timeout=45):
    """Returns a plain-English explanation from the local LLM, or None on
    any failure (Ollama not running, timeout, etc.) so the app can fall
    back to the statistical explanation without crashing."""
    factors_text = "\n".join(
        f"- {FEATURE_DESCRIPTIONS.get(f['feature'], f['feature'])} "
        f"(technical name: {f['feature']}): actual={f['value']}, "
        f"typical_legit={f['typical']}, deviation={f['deviation']} std devs"
        for f in factors
    )
    reliability_note = (
        f"IMPORTANT CONTEXT: at least one of these factors is more than "
        f"{OOD_THRESHOLD_STD} standard deviations from a typical transaction. "
        "That means this transaction contains values far outside anything "
        "the model was trained on -- its prediction should NOT be treated "
        "as trustworthy, regardless of which label it produced. Your "
        "explanation MUST say plainly that this input is far outside the "
        "range of transactions the model has ever seen, so the prediction "
        "may not be reliable. Do not rationalize the prediction as if it "
        "were dependable.\n\n"
        if is_ood else ""
    )
    prompt = (
        "You are explaining a fraud detection model's output to someone "
        "with no data science background. A machine learning model already "
        "made this decision -- you are ONLY explaining its reasons in "
        "plain English. Do not second-guess the prediction, except as "
        "instructed below regarding reliability.\n\n"
        f"{reliability_note}"
        f"Prediction: {prediction}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Top contributing factors (from the model):\n{factors_text}\n\n"
        "Write a short explanation, 3-4 sentences, covering: (1) what the "
        "prediction means overall, (2) each of the top factors above "
        "explained in everyday language -- avoid technical column names "
        "entirely, describe what each factor means and why it's unusual "
        "compared to a typical legitimate transaction. Do not use "
        "underscores, asterisks, or any markdown formatting -- plain "
        "sentences only. Respond ONLY as "
        'JSON: {"explanation": "..."}'
    )
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": {"explanation": {"type": "string"}},
                    "required": ["explanation"],
                },
            },
            timeout=timeout,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return json.loads(content)["explanation"]
    except Exception:
        return None

@st.cache_resource
def load_model():
    model = joblib.load("models/hist_gradient_boosting.joblib")
    return model

@st.cache_data
def load_reference_stats():
    # Used to explain predictions by comparing against typical legit transactions
    X_test = pd.read_pickle("data/X_test.pkl")
    y_test = pd.read_pickle("data/y_test.pkl")
    legit_mask = y_test == 0
    legit_mean = X_test[legit_mask].mean()
    legit_std = X_test[legit_mask].std().replace(0, 1)
    return legit_mean, legit_std, X_test.columns.tolist()

model = load_model()
legit_mean, legit_std, feature_order = load_reference_stats()

st.title("🔍 Financial Fraud Detection")
st.caption("Enter transaction details to check if it looks fraudulent.")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    with col1:
        txn_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
        amount = st.number_input("Amount", min_value=0.0, value=10000.0, step=100.0)
        old_orig = st.number_input("Sender balance BEFORE", min_value=0.0, value=50000.0, step=100.0)
        new_orig = st.number_input("Sender balance AFTER", min_value=0.0, value=40000.0, step=100.0)
    with col2:
        old_dest = st.number_input("Receiver balance BEFORE", min_value=0.0, value=0.0, step=100.0)
        new_dest = st.number_input("Receiver balance AFTER", min_value=0.0, value=10000.0, step=100.0)
        step = st.number_input("Time step (hour index)", min_value=1, value=200, step=1)

    use_llm = st.checkbox(
        "🤖 Generate AI explanation (needs Ollama running locally)",
        value=False,
    )
    submitted = st.form_submit_button("Check Transaction")

if submitted:
    # --- Build the same engineered features used in training ---
    error_orig = old_orig - amount - new_orig
    error_dest = old_dest + amount - new_dest
    orig_emptied = int(old_orig > 0 and new_orig == 0)
    dest_zero_both = int(old_dest == 0 and new_dest == 0 and amount > 0)
    type_transfer = int(txn_type == "TRANSFER")

    row = pd.DataFrame([{
        "step": step,
        "amount": amount,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "errorBalanceOrig": error_orig,
        "errorBalanceDest": error_dest,
        "origAccountEmptied": orig_emptied,
        "destBalanceZeroBoth": dest_zero_both,
        "type_TRANSFER": type_transfer,
    }])[feature_order]  # ensure column order matches training

    proba = model.predict_proba(row)[0][1]
    prediction = "FRAUD" if proba >= 0.5 else "LEGITIMATE"

    # --- Explanation: top deviating features vs typical legit transaction ---
    z_scores = ((row.iloc[0] - legit_mean) / legit_std).abs().sort_values(ascending=False)
    top_factors = [
        {
            "feature": feat,
            "value": round(float(row.iloc[0][feat]), 2),
            "typical": round(float(legit_mean[feat]), 2),
            "deviation": round(float(z), 1),
        }
        for feat, z in z_scores.head(3).items()
    ]
    is_ood = bool(z_scores.iloc[0] > OOD_THRESHOLD_STD)

    st.divider()

    if is_ood:
        st.warning(
            f"⚠️ **Out-of-range input:** one of these values is "
            f"{z_scores.iloc[0]:.0f} standard deviations from a typical "
            "transaction — far beyond anything in the training data. "
            "The prediction below should be treated as unreliable, "
            "regardless of the label shown."
        )

    if prediction == "FRAUD":
        st.error(f"⚠️ Prediction: **{prediction}**  (confidence: {proba:.1%})")
    else:
        st.success(f"✅ Prediction: **{prediction}**  (confidence: {1 - proba:.1%})")

    st.subheader("Why this prediction?")

    llm_text = None
    if use_llm:
        with st.spinner("Asking local LLM for a plain-English explanation..."):
            llm_text = llm_explain(
                prediction,
                proba if prediction == "FRAUD" else 1 - proba,
                top_factors,
                is_ood=is_ood,
            )

    if llm_text:
        st.info(f"🤖 {_clean_llm_text(llm_text)}")
        with st.expander("See underlying statistical factors"):
            for f in top_factors:
                st.write(f"- **{f['feature']}**: value = `{f['value']}`, "
                         f"typical = `{f['typical']}` ({f['deviation']} std devs away)")
    else:
        if use_llm:
            st.caption("⚠️ Couldn't reach local LLM (is Ollama running?) — showing statistical explanation instead.")
        st.write("Top factors compared to a typical legitimate transaction:")
        for f in top_factors:
            st.write(f"- **{f['feature']}**: value = `{f['value']}`, "
                     f"typical = `{f['typical']}` ({f['deviation']} std devs away)")

st.divider()
st.caption(
    "Model: HistGradientBoostingClassifier trained on PaySim simulated "
    "mobile money transactions, filtered to TRANSFER/CASH_OUT types "
    "(the only types where fraud occurs in this dataset)."
)
