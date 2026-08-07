"""
02_preprocess.py — Clean the data and engineer features.

Key decisions made here (be ready to explain these to judges):

1. We keep ONLY TRANSFER and CASH_OUT transactions. EDA (01_eda.py) showed
   fraud never occurs in PAYMENT / CASH_IN / DEBIT. Dropping them removes
   ~3.6M rows of pure noise and lets the model focus on the transaction
   types where fraud is actually possible. This is a real modeling choice,
   not just "we didn't have time" -- say so out loud in your demo.

2. We engineer "balance error" features. In real (and simulated) banking
   data, a legitimate transaction should satisfy:
       oldbalanceOrg - amount = newbalanceOrig
   When this does NOT hold, something inconsistent happened -- which is
   exactly the kind of signal fraud detection lives on. We compute the
   mismatch directly instead of hoping the model discovers it from raw
   balances alone.

3. We drop nameOrig / nameDest (transaction IDs, not predictive) and
   isFlaggedFraud (an existing weak rule-based flag, see 01_eda.py --
   keeping it in would let the model "cheat" off another system rather
   than learn real patterns).

Run: python3 src/02_preprocess.py
"""

import pandas as pd
import numpy as np

DATA_PATH = "data/paysim_dataset.csv"
OUTPUT_PATH = "data/paysim_processed.pkl"

def main():
    print("Loading raw data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Raw shape: {df.shape}")

    # --- Step 1: filter to transaction types where fraud actually occurs ---
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    print(f"After filtering to TRANSFER/CASH_OUT: {df.shape}")
    print(f"Fraud rate in filtered data: {df['isFraud'].mean() * 100:.4f}%")

    # --- Step 2: feature engineering ---

    # Balance mismatch on the sender's side.
    # Should be ~0 for a normal transaction. Large deviations are suspicious.
    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]

    # Balance mismatch on the receiver's side.
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Flag: sender's account emptied out completely by this transaction.
    # A classic fraud pattern -- draining an account fully.
    df["origAccountEmptied"] = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype(int)

    # Flag: receiver had zero balance before AND after -- suspicious for
    # CASH_OUT because it suggests a "mule" account that immediately
    # forwards / withdraws funds rather than holding them.
    df["destBalanceZeroBoth"] = (
        (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)
    ).astype(int)

    # Encode transaction type as binary (1 = TRANSFER, 0 = CASH_OUT)
    df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype(int)

    # --- Step 3: drop columns that don't help prediction ---
    # nameOrig/nameDest: unique-ish IDs, not generalizable patterns
    # isFlaggedFraud: existing weak rule-based flag, see docstring above
    # type: already encoded as type_TRANSFER
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud", "type"]
    df = df.drop(columns=drop_cols)

    print(f"\nFinal feature set: {[c for c in df.columns if c != 'isFraud']}")
    print(f"Final shape: {df.shape}")

    df.to_pickle(OUTPUT_PATH)
    print(f"\nSaved processed data to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
