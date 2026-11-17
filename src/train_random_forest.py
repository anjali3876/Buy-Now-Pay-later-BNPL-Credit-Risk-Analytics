
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load engineered dataset
df = pd.read_csv("data/engineered_credit_data.csv")

# Select financial behaviour features
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

# Transform extreme payment-to-bill ratios
X["payment_to_bill_ratio"] = np.log1p(
    X["payment_to_bill_ratio"].clip(lower=0, upper=10)
)

# Use the same split as Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nRANDOM FOREST MODEL RESULTS")
print("----------------------------------")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["No Default", "Default"]
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nROC-AUC score:")
print(round(roc_auc_score(y_test, probabilities), 4))

# Save model
joblib.dump(model, "outputs/random_forest_model.pkl")

print("\nRandom Forest model saved successfully!")
