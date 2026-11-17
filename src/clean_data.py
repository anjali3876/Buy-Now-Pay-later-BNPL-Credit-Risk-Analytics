import pandas as pd

# Load original dataset
df = pd.read_excel(
    "data/default of credit card clients.xls",
    header=1
)

# Clean column names
df.columns = df.columns.str.strip()

# Rename target column
df = df.rename(columns={
    "default payment next month": "default_next_month"
})

# Remove customer ID
df = df.drop(columns=["ID"])

# Check duplicates before removal
print("Duplicates before cleaning:", df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Check final data quality
print("\nFinal dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicates remaining:", df.duplicated().sum())

# Check default distribution
print("\nDefault distribution:")
print(df["default_next_month"].value_counts())

# Save cleaned dataset
df.to_csv("data/cleaned_credit_data.csv", index=False)

print("\nCleaned dataset saved successfully!")