
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/engineered_credit_data.csv")

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

X["payment_to_bill_ratio"] = np.log1p(
    X["payment_to_bill_ratio"].clip(lower=0, upper=10)
)

# Models
scale = (y == 0).sum() / (y == 1).sum()

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
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
}

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision"
}

print("\nCROSS-VALIDATION RESULTS")
print("----------------------------------")

for name, model in models.items():

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    print(f"\n{name}")
    print("ROC-AUC:", round(results["test_roc_auc"].mean(), 4),
          "+/-", round(results["test_roc_auc"].std(), 4))
    print("Recall:", round(results["test_recall"].mean(), 4))
    print("Precision:", round(results["test_precision"].mean(), 4))

print("\nCross-validation completed!")
