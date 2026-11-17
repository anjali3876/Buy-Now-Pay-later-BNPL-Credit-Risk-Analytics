
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("outputs/random_forest_model.pkl")

# Features must match the model's training features
FEATURES = [
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


def assess_application(
    customer,
    purchase_amount,
    instalments,
    available_monthly_budget
):
    """
    Simulated BNPL application assessment.

    available_monthly_budget means money remaining each month
    after essential expenses and existing commitments.
    """

    # Prepare customer's historical financial features
    X = pd.DataFrame(
        [customer[FEATURES].to_dict()],
        columns=FEATURES
    )

    # Apply the same transformation used during training
    X["payment_to_bill_ratio"] = np.log1p(
        X["payment_to_bill_ratio"].clip(lower=0, upper=10)
    )

    # Predict probability of future default
    default_probability = model.predict_proba(X)[0][1]

    # Assign illustrative risk category
    if default_probability < 0.15:
        risk_category = "Low"
    elif default_probability < 0.30:
        risk_category = "Medium"
    else:
        risk_category = "High"

    # Calculate proposed monthly instalment
    monthly_instalment = purchase_amount / instalments

    # Calculate affordability ratio
    if available_monthly_budget <= 0:
        affordability_ratio = float("inf")
    else:
        affordability_ratio = (
            monthly_instalment / available_monthly_budget
        )

    # Illustrative affordability rules
    if affordability_ratio > 1:
        affordability_status = "Unaffordable"
        recommendation = "DECLINE"
    elif affordability_ratio > 0.30:
        affordability_status = "Requires review"
        recommendation = "REFER FOR REVIEW"
    else:
        affordability_status = "Affordable under demo rule"

        if risk_category == "Low":
            recommendation = "SIMULATED APPROVAL"
        else:
            recommendation = "REFER FOR REVIEW"

    # Return assessment results
    return {
        "default_probability": round(
            float(default_probability) * 100, 2
        ),
        "risk_category": risk_category,
        "purchase_amount": round(purchase_amount, 2),
        "monthly_instalment": round(monthly_instalment, 2),
        "available_monthly_budget": round(
            available_monthly_budget, 2
        ),
        "affordability_ratio": round(
            affordability_ratio * 100, 2
        ) if np.isfinite(affordability_ratio) else None,
        "affordability_status": affordability_status,
        "recommendation": recommendation
    }


# Run a demonstration using a historical customer profile
if __name__ == "__main__":

    df = pd.read_csv("data/engineered_credit_data.csv")

    # Select an example historical profile
    customer = df.iloc[0]

    result = assess_application(
        customer=customer,
        purchase_amount=300,
        instalments=3,
        available_monthly_budget=200
    )

    print("\nBNPL APPLICATION ASSESSMENT")
    print("---------------------------")

    for key, value in result.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("\nEducational prototype only - not a real credit decision.")
