import pandas as pd

# Load customer-level ECL results
df = pd.read_csv("outputs/customer_ecl_results.csv")

# Create risk segments using predicted probability of default
def assign_risk_segment(pd_value):
    if pd_value < 0.15:
        return "Low Risk"
    elif pd_value < 0.30:
        return "Medium Risk"
    elif pd_value < 0.50:
        return "High Risk"
    else:
        return "Very High Risk"

df["Risk_Segment"] = df["PD"].apply(assign_risk_segment)

# Summarise each risk segment
summary = df.groupby("Risk_Segment").agg(
    Customer_Count=("PD", "count"),
    Average_PD=("PD", "mean"),
    Total_ECL=("ECL", "sum"),
    Average_ECL=("ECL", "mean")
).reset_index()

# Order segments logically
risk_order = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
summary["Risk_Segment"] = pd.Categorical(
    summary["Risk_Segment"],
    categories=risk_order,
    ordered=True
)
summary = summary.sort_values("Risk_Segment")

# Display results
print("\nCUSTOMER RISK SEGMENTATION")
print("=" * 65)
print(summary.to_string(index=False))

# Save results
summary.to_csv("outputs/risk_segmentation_summary.csv", index=False)
df.to_csv("outputs/customer_risk_segments.csv", index=False)

print("\nRisk segmentation completed successfully.")
print("Files saved in the outputs folder.")