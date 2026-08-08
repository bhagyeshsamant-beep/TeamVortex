"""
05_llm_explain.py — Generate natural-language fraud explanations using a
local LLM (Bonsai 27B via Ollama), on top of the existing statistical
explanation from 04_explain.py.

WHY THIS EXISTS (for your presentation):
  04_explain.py already gives a correct, defensible explanation: which
  features deviated most from a "normal" transaction, in standard
  deviations. That's good for YOU to justify the model, but it's clunky
  for an end user (e.g. a fraud analyst) to read at a glance.

  This script takes that same statistical output and asks a local LLM to
  turn it into one readable sentence. The LLM is NOT making the fraud
  decision and is NOT looking at raw data -- it only rephrases numbers
  your trained model already produced. This keeps the "AI/ML Integration"
  and "Explainability" criteria solid: the classifier decides, the LLM
  explains.

SETUP (run once, on your own machine, not in this sandbox):
    1. Install Ollama: https://ollama.com/download
    2. Pull the model:
         ollama pull hf.co/prism-ml/Ternary-Bonsai-27B-gguf
       (5.9GB, better quality -- fits fine in 16GB RAM)
       If that's too slow on your GTX 1650, fall back to something much
       lighter and still solid for this task:
         ollama pull phi4-mini
       and change MODEL_NAME below.
    3. Make sure the Ollama app/service is running (it starts a local
       server at http://localhost:11434 automatically).

Run: python3 src/05_llm_explain.py
"""

import pandas as pd
import numpy as np
import joblib
import requests
import json

MODEL_NAME = "hf.co/prism-ml/Ternary-Bonsai-27B-gguf"  # swap to "phi4-mini" if too slow
OLLAMA_URL = "http://localhost:11434/api/chat"

def get_statistical_factors(row, legit_mean, legit_std, top_n=3):
    """Same z-score approach as 04_explain.py -- the ground-truth reasons."""
    z_scores = ((row - legit_mean) / legit_std).abs().sort_values(ascending=False)
    factors = []
    for feat, z in z_scores.head(top_n).items():
        factors.append({
            "feature": feat,
            "value": round(float(row[feat]), 2),
            "typical": round(float(legit_mean[feat]), 2),
            "deviation_std": round(float(z), 1),
        })
    return factors

def llm_explain(prediction, confidence, factors, timeout=30):
    """
    Ask the local LLM to turn statistical factors into one plain-English
    sentence. Returns None on any failure so the caller can fall back to
    the raw statistical explanation -- this must never crash the app.
    """
    factors_text = "\n".join(
        f"- {f['feature']}: actual={f['value']}, typical_legit={f['typical']}, "
        f"deviation={f['deviation_std']} std devs"
        for f in factors
    )

    prompt = (
        "You are explaining a fraud detection model's output to a bank "
        "analyst. A machine learning model already made this decision -- "
        "you are ONLY rephrasing its reasons in plain English. Do not "
        "second-guess the prediction.\n\n"
        f"Prediction: {prediction}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Top contributing factors (from the model):\n{factors_text}\n\n"
        "Write ONE short sentence (max 30 words) explaining why this "
        "transaction was flagged, in plain English, referencing the "
        "specific factors above. Respond ONLY as JSON: "
        '{"explanation": "..."}'
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
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
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return json.loads(content)["explanation"]
    except Exception as e:
        print(f"[LLM explain skipped: {e}]")
        return None

def main():
    print("Loading model and test data...")
    model = joblib.load("models/hist_gradient_boosting.joblib")
    X_test = pd.read_pickle("data/X_test.pkl")
    y_test = pd.read_pickle("data/y_test.pkl")

    legit_mask = y_test == 0
    legit_mean = X_test[legit_mask].mean()
    legit_std = X_test[legit_mask].std().replace(0, 1)

    y_proba = model.predict_proba(X_test)[:, 1]
    fraud_indices = y_test[y_test == 1].index[:3]

    print(f"\nUsing local LLM: {MODEL_NAME}")
    print("(Make sure Ollama is running -- `ollama serve` or the desktop app)\n")

    for idx in fraud_indices:
        row = X_test.loc[idx]
        confidence = y_proba[X_test.index.get_loc(idx)]
        factors = get_statistical_factors(row, legit_mean, legit_std)

        print("=" * 60)
        print(f"Transaction (index {idx}) -- confidence: {confidence:.1%}")
        print("Statistical factors:")
        for f in factors:
            print(f"  - {f['feature']}: {f['value']} vs typical {f['typical']} "
                  f"({f['deviation_std']} std)")

        explanation = llm_explain("FRAUD", confidence, factors)
        if explanation:
            print(f"\nLLM explanation: {explanation}")
        else:
            print("\nLLM explanation: [unavailable -- falling back to statistical "
                  "factors above, which is exactly what the app.py UI will do too]")
        print()

if __name__ == "__main__":
    main()
