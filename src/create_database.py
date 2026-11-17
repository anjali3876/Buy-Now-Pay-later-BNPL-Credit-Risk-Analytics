import pandas as pd
import sqlite3

# Load cleaned dataset
df = pd.read_csv("data/cleaned_credit_data.csv")

# Connect to SQLite database
connection = sqlite3.connect(
    "database/bnpl_risk.db"
)

# Import data into SQL table
df.to_sql(
    "customers",
    connection,
    if_exists="replace",
    index=False
)

# SQL analysis 1: Default rate
query1 = """
SELECT
    COUNT(*) AS total_customers,
    SUM(default_next_month) AS defaulted_customers,
    ROUND(AVG(default_next_month) * 100, 2) AS default_rate
FROM customers;
"""

print("\nOverall default analysis:")
print(pd.read_sql_query(query1, connection))


# SQL analysis 2: Average credit limit by default status
query2 = """
SELECT
    default_next_month,
    COUNT(*) AS customer_count,
    ROUND(AVG(LIMIT_BAL), 2) AS average_credit_limit
FROM customers
GROUP BY default_next_month;
"""

print("\nCredit limit analysis:")
print(pd.read_sql_query(query2, connection))


# SQL analysis 3: Customers grouped by age
query3 = """
SELECT
    CASE
        WHEN AGE < 25 THEN 'Under 25'
        WHEN AGE BETWEEN 25 AND 40 THEN '25-40'
        WHEN AGE BETWEEN 41 AND 60 THEN '41-60'
        ELSE 'Over 60'
    END AS age_group,
    COUNT(*) AS customer_count,
    ROUND(AVG(default_next_month) * 100, 2) AS default_rate
FROM customers
GROUP BY age_group
ORDER BY default_rate DESC;
"""

print("\nAge group risk analysis:")
print(pd.read_sql_query(query3, connection))

connection.close()