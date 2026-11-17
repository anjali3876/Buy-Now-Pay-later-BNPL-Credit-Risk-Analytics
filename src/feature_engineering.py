
import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/cleaned_credit_data.csv")

# Define historical repayment, bill and payment columns
repayment_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
payment_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

# 1. Credit utilisation
df["credit_utilisation"] = (
    df["BILL_AMT1"].clip(lower=0)
    / df["LIMIT_BAL"].replace(0, np.nan)
).fillna(0)

# 2. Number of months with delayed repayments
df["total_delayed_months"] = (df[repayment_cols] > 0).sum(axis=1)

# 3. Most severe historical repayment status
df["max_payment_delay"] = df[repayment_cols].max(axis=1)

# 4. Average historical bill amount
df["average_bill"] = df[bill_cols].mean(axis=1)

# 5. Average historical payment amount
df["average_payment"] = df[payment_cols].mean(axis=1)

# 6. Overall payment-to-bill ratio
total_payments = df[payment_cols].sum(axis=1)

positive_bills = df[bill_cols].clip(lower=0).sum(axis=1)

df["payment_to_bill_ratio"] = (
    total_payments / positive_bills.replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0)

# 7. Change in bill amount over the historical period
df["bill_amount_change"] = df["BILL_AMT1"] - df["BILL_AMT6"]

# 8. Change in payment amount over the historical period
df["payment_amount_change"] = df["PAY_AMT1"] - df["PAY_AMT6"]

# Save engineered dataset
df.to_csv("data/engineered_credit_data.csv", index=False)

print("Feature engineering completed!")
print("Dataset shape:", df.shape)

print("\nNew financial features:")
print([
    "credit_utilisation",
    "total_delayed_months",
    "max_payment_delay",
    "average_bill",
    "average_payment",
    "payment_to_bill_ratio",
    "bill_amount_change",
    "payment_amount_change"
])

print("\nFeature summary:")
print(df[[
    "credit_utilisation",
    "total_delayed_months",
    "max_payment_delay",
    "payment_to_bill_ratio"
]].describe())
