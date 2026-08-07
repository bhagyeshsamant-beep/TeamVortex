"""
03_train_model.py — Train and compare fraud detection models.

We train TWO models on purpose:
  1. Logistic Regression  -> simple, interpretable baseline
  2. HistGradientBoosting  -> stronger model, similar family to XGBoost/LightGBM,
     built into scikit-learn (no extra install needed -- this matters for
     judges who clone your repo and run `pip install -r requirements.txt`;
     nothing breaks).

Class imbalance handling: since fraud is ~0.3% of this filtered data, we use
class_weight="balanced" on both models. This tells the model to treat each
fraud example as worth much more than a legitimate one during training,
instead of letting it get away with just predicting "not fraud" every time.
This is a standard, well-understood technique for imbalanced classification.

We evaluate with Precision, Recall, F1, and ROC-AUC -- NOT accuracy, because
accuracy is meaningless on this data (see 01_eda.py).

Run: python3 src/03_train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import joblib

DATA_PATH = "data/paysim_processed.pkl"

def evaluate(name, y_test, y_pred, y_proba):
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {name}")
    print(f"{'=' * 60}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion matrix:")
    print("             Predicted Legit  Predicted Fraud")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Actual Legit   {cm[0][0]:>10,}      {cm[0][1]:>10,}")
    print(f"Actual Fraud   {cm[1][0]:>10,}      {cm[1][1]:>10,}")
    print("\nFull report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

def main():
    print("Loading processed data...")
    df = pd.read_pickle(DATA_PATH)

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    # Stratified split -- crucial for imbalanced data, keeps the same fraud
    # ratio in both train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Fraud in train: {y_train.sum()}, Fraud in test: {y_test.sum()}")

    # --- Model 1: Logistic Regression (baseline) ---
    # Logistic regression is scale-sensitive, so we standardize features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Logistic Regression (baseline)...")
    log_reg = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    log_reg.fit(X_train_scaled, y_train)

    y_pred_lr = log_reg.predict(X_test_scaled)
    y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
    evaluate("Logistic Regression (baseline)", y_test, y_pred_lr, y_proba_lr)

    # --- Model 2: HistGradientBoostingClassifier (main model) ---
    print("\nTraining HistGradientBoostingClassifier (main model)...")
    hgb = HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42, max_iter=200
    )
    hgb.fit(X_train, y_train)  # tree models don't need scaling

    y_pred_hgb = hgb.predict(X_test)
    y_proba_hgb = hgb.predict_proba(X_test)[:, 1]
    evaluate("HistGradientBoosting (main model)", y_test, y_pred_hgb, y_proba_hgb)

    # --- Save everything needed for the app / evaluation script ---
    joblib.dump(log_reg, "models/logistic_regression.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(hgb, "models/hist_gradient_boosting.joblib")
    X_test.to_pickle("data/X_test.pkl")
    y_test.to_pickle("data/y_test.pkl")
    print("\nModels saved to models/. Test set saved to data/ for reuse.")

if __name__ == "__main__":
    main()
