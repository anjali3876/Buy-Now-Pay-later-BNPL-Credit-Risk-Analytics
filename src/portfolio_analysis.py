import pandas as pd

# Load customer-level risk results
df = pd.read_csv("outputs/customer_ecl_results.csv")

# Load risk segmentation results
segments = pd.read_csv("outputs/risk_segmentation_summary.csv")

# Portfolio metrics
total_customers = len(df)
total_ead = df["EAD"].sum()
total_ecl = df["ECL"].sum()
average_pd = df["PD"].mean()
actual_default_rate = df["actual_default"].mean()

# Risk concentration
very_high_risk_ecl = segments.loc[
    segments["Risk_Segment"] == "Very High Risk", "Total_ECL"
].sum()

very_high_risk_share = (
    very_high_risk_ecl / total_ecl * 100 if total_ecl else 0
)

# Portfolio summary
print("\nPORTFOLIO RISK ANALYSIS")
print("=" * 55)
print(f"Total customers:             {total_customers:,}")
print(f"Total exposure (EAD):        {total_ead:,.2f}")
print(f"Total expected credit loss:  {total_ecl:,.2f}")
print(f"Average predicted PD:        {average_pd:.2%}")
print(f"Actual default rate:         {actual_default_rate:.2%}")
print(f"Very high risk ECL share:    {very_high_risk_share:.2f}%")

# Save portfolio summary
portfolio_summary = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Total EAD",
        "Total ECL",
        "Average Predicted PD",
        "Actual Default Rate",
        "Very High Risk ECL Share"
    ],
    "Value": [
        total_customers,
        total_ead,
        total_ecl,
        average_pd,
        actual_default_rate,
        very_high_risk_share
    ]
})

portfolio_summary.to_csv(
    "outputs/portfolio_summary.csv", index=False
)

# Save risk segment analysis
segments.to_csv(
    "outputs/portfolio_risk_segments.csv", index=False
)

print("\nPortfolio analysis completed successfully.")
print("Results saved in the outputs folder.")