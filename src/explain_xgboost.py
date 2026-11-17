
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Load engineered dataset
df = pd.read_csv("data/engineered_credit_data.csv")

# Use exactly the same features as the XGBoost training script
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

# Apply the same transformation
X["payment_to_bill_ratio"] = np.log1p(
    X["payment_to_bill_ratio"].clip(lower=0, upper=10)
)

# Recreate the same test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained XGBoost model
model = joblib.load("outputs/xgboost_model.pkl")

# Calculate SHAP values
print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(
    "outputs/xgboost_shap_summary.png",
    bbox_inches="tight",
    dpi=300
)
plt.close()

# Feature importance bar plot
plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(
    "outputs/xgboost_shap_importance.png",
    bbox_inches="tight",
    dpi=300
)
plt.close()

print("\nSHAP analysis completed!")
print("Saved:")
print("outputs/xgboost_shap_summary.png")
print("outputs/xgboost_shap_importance.png")
