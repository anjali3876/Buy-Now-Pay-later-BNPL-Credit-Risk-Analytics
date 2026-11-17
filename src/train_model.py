
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load engineered dataset
df = pd.read_csv("data/engineered_credit_data.csv")

# Define financial behaviour features
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

# Prepare model inputs
X = df[features].copy()
y = df["default_next_month"]

# Reduce the influence of extreme values in the ratio
X["payment_to_bill_ratio"] = (
    np.log1p(X["payment_to_bill_ratio"].clip(lower=0, upper=10))
)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build Logistic Regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluate model
print("\nLOGISTIC REGRESSION MODEL RESULTS")
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

# Save trained model
os.makedirs("outputs", exist_ok=True)

import joblib
joblib.dump(model, "outputs/logistic_regression_model.pkl")

print("\nModel saved successfully!")
