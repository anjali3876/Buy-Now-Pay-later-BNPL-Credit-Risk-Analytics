
import pandas as pd
import matplotlib.pyplot as plt

# Cross-validation results calculated earlier
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "ROC_AUC": [
        0.7494,
        0.7831,
        0.7801
    ],
    "Default_Recall": [
        0.6100,
        0.6024,
        0.6317
    ],
    "Default_Precision": [
        None,
        0.4936,
        0.4649
    ]
})

print("\nMODEL PERFORMANCE COMPARISON")
print("=" * 65)
print(results.to_string(index=False))

# Save comparison results
results.to_csv("outputs/model_comparison.csv", index=False)

# ROC-AUC chart
plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["ROC_AUC"])
plt.title("Model Comparison - ROC-AUC")
plt.ylabel("ROC-AUC")
plt.ylim(0.65, 0.85)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

print("\nModel comparison saved successfully.")
