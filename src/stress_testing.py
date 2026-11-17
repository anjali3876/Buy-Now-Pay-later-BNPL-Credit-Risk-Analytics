
import pandas as pd
import matplotlib.pyplot as plt

# Load existing ECL results
df = pd.read_csv("outputs/customer_ecl_results.csv")

# Define illustrative economic scenarios
scenarios = {
    "Base Case": {
        "pd_multiplier": 1.0,
        "lgd": 0.45,
        "ead_multiplier": 1.0
    },
    "Moderate Stress": {
        "pd_multiplier": 1.25,
        "lgd": 0.55,
        "ead_multiplier": 1.05
    },
    "Severe Stress": {
        "pd_multiplier": 1.50,
        "lgd": 0.65,
        "ead_multiplier": 1.10
    }
}

results = []

# Calculate ECL under each scenario
for scenario, assumptions in scenarios.items():

    stressed_pd = (
        df["PD"] * assumptions["pd_multiplier"]
    ).clip(upper=1.0)

    stressed_lgd = assumptions["lgd"]

    stressed_ead = (
        df["EAD"] * assumptions["ead_multiplier"]
    )

    stressed_ecl = (
        stressed_pd * stressed_lgd * stressed_ead
    )

    results.append({
        "Scenario": scenario,
        "Average PD": stressed_pd.mean(),
        "LGD": stressed_lgd,
        "Total EAD": stressed_ead.sum(),
        "Total ECL": stressed_ecl.sum(),
        "Average ECL": stressed_ecl.mean()
    })

# Create summary table
summary = pd.DataFrame(results)

print("\nCREDIT RISK STRESS TESTING")
print("=" * 60)

print(summary.to_string(
    index=False,
    formatters={
        "Average PD": "{:.2%}".format,
        "LGD": "{:.2%}".format,
        "Total EAD": "{:,.2f}".format,
        "Total ECL": "{:,.2f}".format,
        "Average ECL": "{:,.2f}".format
    }
))

# Save results
summary.to_csv(
    "outputs/stress_testing_results.csv",
    index=False
)

# Create ECL comparison chart
plt.figure(figsize=(9, 6))

plt.bar(
    summary["Scenario"],
    summary["Total ECL"]
)

plt.title("Expected Credit Loss Under Economic Stress")
plt.xlabel("Economic Scenario")
plt.ylabel("Total Expected Credit Loss")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/stress_testing_chart.png",
    dpi=300
)

plt.close()

print("\nStress testing results saved successfully.")
print("Chart saved to outputs/stress_testing_chart.png")
