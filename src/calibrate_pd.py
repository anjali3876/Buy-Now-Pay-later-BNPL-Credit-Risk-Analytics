
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scale = (y_train == 0).sum() / (y_train == 1).sum()

base_model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

# Calibrate using a separate calibration split
X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

base_model.fit(X_fit, y_fit)

calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(base_model),
    method="sigmoid"
)

calibrated_model.fit(X_cal, y_cal)

# Predict probabilities on untouched test set
probabilities = calibrated_model.predict_proba(X_test)[:, 1]

uncalibrated_probabilities = base_model.predict_proba(X_test)[:, 1]

print("\nUNCALIBRATED BRIER SCORE:",
      round(brier_score_loss(y_test, uncalibrated_probabilities), 4))

print("CALIBRATED BRIER SCORE:",
      round(brier_score_loss(y_test, probabilities), 4))
uncalibrated_probabilities = base_model.predict_proba(X_test)[:, 1]

print("Uncalibrated Brier score:",
      round(brier_score_loss(y_test, uncalibrated_probabilities), 4))

print("Calibrated Brier score:",
      round(brier_score_loss(y_test, probabilities), 4))
print("\nPD CALIBRATION RESULTS")
print("----------------------------------")

print("Actual default rate:",
      round(y_test.mean(), 4))

print("Average predicted PD:",
      round(probabilities.mean(), 4))

print("Brier score:",
      round(brier_score_loss(y_test, probabilities), 4))

# Calibration plot
actual, predicted = calibration_curve(
    y_test,
    probabilities,
    n_bins=10,
    strategy="quantile"
)

plt.figure(figsize=(8, 6))
plt.plot(predicted, actual, marker="o", label="Calibrated XGBoost")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

plt.xlabel("Mean predicted probability")
plt.ylabel("Actual default rate")
plt.title("XGBoost Probability Calibration")
plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/pd_calibration.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nCalibration plot saved to outputs/pd_calibration.png")
