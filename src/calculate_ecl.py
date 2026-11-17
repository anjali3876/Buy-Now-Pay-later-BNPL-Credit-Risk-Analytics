
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

# Load data
df = pd.read_csv("data/engineered_credit_data.csv")

# Features used by XGBoost
features = [
    "LIMIT_BAL",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "credit_utilisation",
    "total_delayed_months",
    "max_payment_delay",
    "average_bill",
    "average_payment",
    "payment_to_bill_ratio",
    "bill_amount_change",
    "payment_amount_change"
]

X = df[features].copy()
y = df["default_next_month"]

# Same feature transformation as XGBoost training
X["payment_to_bill_ratio"] = np.log1p(
    X["payment_to_bill_ratio"].clip(lower=0, upper=10)
)

# Generate out-of-fold PD predictions
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

pd_predictions = np.zeros(len(df))

print("Generating out-of-fold default probabilities...")

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx]

    scale = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pd_predictions[test_idx] = model.predict_proba(X_test)[:, 1]

    print(f"Fold {fold}/5 completed")

# Calculate credit-risk components
LGD = 0.45

EAD = df["BILL_AMT1"].clip(lower=0)

PD = pd.Series(pd_predictions, index=df.index)

ECL = PD * LGD * EAD

# Create results
results = pd.DataFrame({
    "PD": PD,
    "LGD": LGD,
    "EAD": EAD,
    "ECL": ECL,
    "actual_default": y
})

# Save customer-level results
results.to_csv("outputs/customer_ecl_results.csv", index=False)

# Portfolio summary
print("\nEXPECTED CREDIT LOSS SUMMARY")
print("----------------------------------")

print("Customers:", len(results))
print("Average PD:", round(PD.mean(), 4))
print("Illustrative LGD:", LGD)
print("Total EAD:", round(EAD.sum(), 2))
print("Total Expected Credit Loss:", round(ECL.sum(), 2))
print("Average ECL per customer:", round(ECL.mean(), 2))

print("\nECL calculation completed!")
print("Saved to outputs/customer_ecl_results.csv")
