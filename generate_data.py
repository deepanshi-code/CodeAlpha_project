"""Generate the synthetic datasets used by both notebooks.

The labels are produced by hand-written rules (see each function), so they are
a deterministic function of the features. This makes the data useful for
practising the ML workflow, but model scores on it do NOT reflect real-world
credit-risk or disease-prediction performance.

Usage:  python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
N_SAMPLES = 5000
SEED = 42


def make_credit_data(n=N_SAMPLES, seed=SEED):
    rng = np.random.RandomState(seed)
    income = rng.randint(20000, 150000, n)
    debts = rng.randint(0, 80000, n)
    payment_history = rng.randint(0, 2, n)
    loan_amount = rng.randint(5000, 50000, n)
    age = rng.randint(18, 65, n)

    # Labelling rule: a weighted score thresholded at 50
    credit_score = (income / 1000 - debts / 2000 + payment_history * 20
                    - loan_amount / 5000 + age / 10)
    creditworthy = (credit_score > 50).astype(int)

    return pd.DataFrame({
        "Income": income,
        "Debts": debts,
        "Payment_History": payment_history,
        "Loan_Amount": loan_amount,
        "Age": age,
        "Creditworthy": creditworthy,
    })


def make_medical_data(n=N_SAMPLES, seed=SEED):
    rng = np.random.RandomState(seed)
    glucose = rng.randint(70, 200, n)
    blood_pressure = rng.randint(60, 140, n)
    bmi = rng.uniform(18, 40, n)
    age = rng.randint(18, 80, n)
    insulin = rng.randint(15, 276, n)

    # Labelling rule: positive if at least 3 of 5 risk thresholds are exceeded
    risk_factors = ((glucose > 125).astype(int) + (blood_pressure > 90).astype(int)
                    + (bmi > 30).astype(int) + (age > 45).astype(int)
                    + (insulin > 150).astype(int))
    outcome = (risk_factors >= 3).astype(int)

    return pd.DataFrame({
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "BMI": bmi,
        "Age": age,
        "Insulin": insulin,
        "Outcome": outcome,
    })


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    make_credit_data().to_csv(DATA_DIR / "credit_data.csv", index=False)
    make_medical_data().to_csv(DATA_DIR / "medical_data.csv", index=False)
    print(f"Wrote credit_data.csv and medical_data.csv to {DATA_DIR}")
