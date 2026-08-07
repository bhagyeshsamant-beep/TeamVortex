"""
01_eda.py — Exploratory Data Analysis on the PaySim dataset.

Goal: understand the data BEFORE building any model. This is not optional —
judges will ask "why did you make this modeling decision" and "we looked at
the data and saw X" is a much stronger answer than "we just tried stuff."

Run: python3 src/01_eda.py
"""

import pandas as pd

DATA_PATH = "data/paysim_dataset.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("BASIC SHAPE")
    print("=" * 60)
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(df.columns.tolist())

    print("\n" + "=" * 60)
    print("FRAUD DISTRIBUTION (the class imbalance problem)")
    print("=" * 60)
    fraud_counts = df["isFraud"].value_counts()
    fraud_pct = df["isFraud"].mean() * 100
    print(fraud_counts)
    print(f"Fraud rate: {fraud_pct:.4f}%  <-- extremely imbalanced")
    print("This is WHY accuracy alone is a useless metric here.")
    print("A model that predicts 'not fraud' for everything would still be")
    print(f"{100 - fraud_pct:.2f}% 'accurate' while catching zero fraud.")

    print("\n" + "=" * 60)
    print("TRANSACTION TYPES")
    print("=" * 60)
    print(df["type"].value_counts())

    print("\n" + "=" * 60)
    print("FRAUD COUNT BY TRANSACTION TYPE (key finding)")
    print("=" * 60)
    fraud_by_type = df.groupby("type")["isFraud"].sum()
    print(fraud_by_type)
    print("\n--> Fraud ONLY occurs in TRANSFER and CASH_OUT transactions.")
    print("    PAYMENT, CASH_IN, DEBIT have zero fraud cases in this dataset.")
    print("    This matches real-world logic: fraud requires moving money")
    print("    OUT of an account (transfer to another account, or cashing out)")
    print("    -- you can't commit this kind of fraud by receiving a payment.")

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())
    print("(none expected in the standard PaySim dataset)")

    print("\n" + "=" * 60)
    print("isFlaggedFraud (existing rule-based system, NOT our model)")
    print("=" * 60)
    print(df["isFlaggedFraud"].value_counts())
    print("This is an existing business-rule flag (fires on large TRANSFERs).")
    print("It fires only 16 times total, so it's not a useful feature on its")
    print("own -- but it tells you PaySim's own simulated rule system is very")
    print("weak, which is a nice line for your presentation: 'the existing")
    print("rule-based flag catches almost nothing; here's what ML adds.'")

if __name__ == "__main__":
    main()
