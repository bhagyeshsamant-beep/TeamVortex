"""
04_explain.py — Explainability for the fraud model.

Judges explicitly evaluate "Explainability" (see the crop problem statement's
evaluation criteria, and it's good practice here too) -- a model that just
says "FRAUD" with no reason is much weaker than one that also says why.

Two levels of explanation:

1. GLOBAL: which features matter most across ALL predictions.
   We use sklearn's permutation_importance -- it works with any model
   (not just tree models with .feature_importances_), by measuring how
   much performance drops when a feature's values are randomly shuffled.
   Bigger drop = more important feature.

2. PER-TRANSACTION: for a single flagged transaction, we compare its
   feature values against the average LEGITIMATE transaction, and report
   which features deviate the most (in standard deviations). This gives
   a plain-English reason like "amount is 8.2x higher than typical" for
   each flagged transaction -- simple, fast, and easy to explain to judges
   without needing extra libraries.

   Note: if you have internet access on your own laptop, `pip install shap`
   and swap in shap.TreeExplainer for a more rigorous version of this same
   idea (SHAP values instead of z-scores). The code below is a dependency-free
   substitute that works everywhere, including offline demo environments.

Run: python3 src/04_explain.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.inspection import permutation_importance

def main():
    print("Loading model and test data...")
    model = joblib.load("models/hist_gradient_boosting.joblib")
    X_test = pd.read_pickle("data/X_test.pkl")
    y_test = pd.read_pickle("data/y_test.pkl")

    # --- 1. Global feature importance ---
    print("\nComputing permutation importance (this takes a moment)...")
    # Use a sample for speed -- 20k rows is plenty to rank importance reliably
    sample_idx = X_test.sample(n=min(20000, len(X_test)), random_state=42).index
    X_sample = X_test.loc[sample_idx]
    y_sample = y_test.loc[sample_idx]

    result = permutation_importance(
        model, X_sample, y_sample, n_repeats=5, random_state=42, scoring="f1"
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": result.importances_mean
    }).sort_values("importance", ascending=False)

    print("\n" + "=" * 60)
    print("GLOBAL FEATURE IMPORTANCE (which features matter most overall)")
    print("=" * 60)
    print(importance_df.to_string(index=False))
    importance_df.to_csv("outputs/feature_importance.csv", index=False)

    # --- 2. Per-transaction explanation ---
    print("\n" + "=" * 60)
    print("PER-TRANSACTION EXPLANATION (example on 3 flagged fraud cases)")
    print("=" * 60)

    # Build a "normal" profile from legitimate transactions to compare against
    legit_mask = y_test == 0
    legit_mean = X_test[legit_mask].mean()
    legit_std = X_test[legit_mask].std().replace(0, 1)  # avoid divide-by-zero

    y_proba = model.predict_proba(X_test)[:, 1]
    fraud_indices = y_test[y_test == 1].index[:3]  # 3 example fraud cases

    for idx in fraud_indices:
        row = X_test.loc[idx]
        confidence = y_proba[X_test.index.get_loc(idx)]
        z_scores = ((row - legit_mean) / legit_std).abs().sort_values(ascending=False)
        top_reasons = z_scores.head(3)

        print(f"\nTransaction (index {idx}):")
        print(f"  Fraud confidence: {confidence:.2%}")
        print(f"  Top contributing factors (deviation from normal transactions):")
        for feat, z in top_reasons.items():
            actual_val = row[feat]
            typical_val = legit_mean[feat]
            print(f"    - {feat}: value={actual_val:.2f} vs typical={typical_val:.2f} "
                  f"({z:.1f} std deviations away)")

    print("\nSaved global importance ranking to outputs/feature_importance.csv")

if __name__ == "__main__":
    main()
