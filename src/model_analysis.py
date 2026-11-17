
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Load trained Random Forest model
model = joblib.load("outputs/random_forest_model.pkl")

# Feature names used during training
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

# Extract feature importance
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

# Display top 10 features
print("\nTOP 10 FEATURES INFLUENCING MODEL PREDICTIONS")
print(importance.head(10).to_string(index=False))

# Save results
os.makedirs("outputs", exist_ok=True)

importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

# Create feature importance chart
top_features = importance.head(10).sort_values("importance")

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title("Random Forest - Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.close()

print("\nFeature importance analysis saved successfully!")
