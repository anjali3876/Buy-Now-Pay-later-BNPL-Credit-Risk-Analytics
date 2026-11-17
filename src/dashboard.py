import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
st.set_option("client.toolbarMode", "hidden")

from decision_engine import assess_application


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="BNPL Credit Risk Analytics",
    page_icon="💳",
    layout="wide"
)


# ==========================================
# SHARED INTERFACE STYLING
# ==========================================

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #F0F4F8;
    border: 1px solid #D9E2EC;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #486581;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #142B4A;
    font-weight: 700;
}

.dashboard-anchor-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0 0 20px 0;
    padding: 10px;
    background: #F0F4F8;
    border: 1px solid #D9E2EC;
    border-radius: 10px;
}

.dashboard-anchor-nav a {
    display: inline-block;
    padding: 7px 12px;
    color: #142B4A !important;
    background: #FFFFFF;
    border: 1px solid #BCCCDC;
    border-radius: 7px;
    text-decoration: none !important;
    font-weight: 600;
}

.dashboard-anchor-nav a:hover {
    background: #D9EAF7;
    border-color: #627D98;
}
</style>
""", unsafe_allow_html=True)


def render_anchor_nav(items):
    """Render a horizontal row of links to anchors within the current section."""
    links = "".join(
        f'<a href="#{anchor_id}">{label}</a>'
        for label, anchor_id in items
    )
    st.markdown(
        f'<nav class="dashboard-anchor-nav">{links}</nav>',
        unsafe_allow_html=True
    )


def render_anchor(anchor_id):
    """Create a matching target for an in-section navigation link."""
    st.markdown(
        f'<div id="{anchor_id}"></div>',
        unsafe_allow_html=True
    )


# ==========================================
# LOAD PROJECT DATA
# ==========================================

@st.cache_data
@st.cache_data
def load_data():
    ecl = pd.read_csv("outputs/customer_ecl_results.csv")
    segments = pd.read_csv("outputs/risk_segmentation_summary.csv")
    stress = pd.read_csv("outputs/stress_testing_results.csv")
    portfolio = pd.read_csv("outputs/portfolio_summary.csv")
    customers = pd.read_csv("outputs/customer_risk_segments.csv")
    comparison = pd.read_csv("outputs/model_comparison.csv")

    engineered = pd.read_csv("data/engineered_credit_data.csv")
    repayment = pd.read_csv("data/cleaned_credit_data.csv")

    return (
        ecl, segments, stress, portfolio,
        customers, comparison, engineered, repayment
    )


(
    ecl, segments, stress, portfolio,
    customers, comparison, engineered, repayment
) = load_data()


# ==========================================
# PORTFOLIO METRICS
# ==========================================


# ==========================================
# PORTFOLIO METRICS
# ==========================================

total_customers = len(ecl)
total_ead = ecl["EAD"].sum()
total_ecl = ecl["ECL"].sum()
average_pd = ecl["PD"].mean()
default_rate = ecl["actual_default"].mean()


def create_excel_report():
    report = BytesIO()

    summary = pd.DataFrame({
        "Metric": [
            "Total Customers",
            "Total EAD",
            "Total ECL",
            "Average Predicted PD",
            "Actual Default Rate",
        ],
        "Value": [
            total_customers,
            total_ead,
            total_ecl,
            average_pd,
            default_rate,
        ],
    })

    customer_risk_summary = (
        customers.groupby("Risk_Segment")
        .size()
        .reset_index(name="Customer_Count")
    )

    ecl_risk_summary = (
        customers.groupby("Risk_Segment")
        .agg(
            Customer_Count=("ECL", "size"),
            Total_ECL=("ECL", "sum"),
        )
        .reset_index()
    )

    portfolio_analysis = pd.DataFrame({
        "Metric": ["Total EAD", "Average PD", "Default Rate"],
        "Value": [
            customers["EAD"].sum(),
            customers["PD"].mean(),
            customers["actual_default"].mean(),
        ],
    })

    with pd.ExcelWriter(report, engine="openpyxl") as writer:
        summary.to_excel(
            writer, sheet_name="Portfolio KPIs", index=False
        )
        segments.to_excel(
            writer, sheet_name="Risk Segments", index=False
        )
        stress.to_excel(
            writer, sheet_name="Stress Tests", index=False
        )
        comparison.to_excel(
            writer, sheet_name="Model Performance", index=False
        )
        customer_risk_summary.to_excel(
            writer, sheet_name="Customer Risk Analysis", index=False
        )
        ecl_risk_summary.to_excel(
            writer, sheet_name="ECL Analysis", index=False
        )
        portfolio_analysis.to_excel(
            writer, sheet_name="Portfolio Analysis", index=False
        )

    return report.getvalue()
# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("💳 BNPL Analytics")
st.sidebar.caption("Credit Risk Management Platform")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Portfolio Overview",
        "Customer Risk",
        "Stress Testing",
        "Model Performance",
        "Model Explainability",
        "Customer Application Assessment"
    ]
)

st.sidebar.divider()

st.sidebar.caption("Quantitative Risk Analytics")
st.sidebar.caption("Educational prototype")


# ==========================================
# MAIN DASHBOARD HEADING
# ==========================================

st.markdown("""
<div style="
    background: linear-gradient(120deg, #142B4A, #294F78);
    padding: 28px;
    border-radius: 12px;
    margin-bottom: 24px;
">
    <h1 style="color: white; margin: 0;">
        BNPL Credit Risk Analytics
    </h1>
    <p style="color: #DCE6F1; margin: 8px 0 0 0;">
        Quantitative Analytics | Portfolio Monitoring |
        Credit Risk Modelling
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Educational prototype using historical public credit data. "
    "Not a real lending decision or validated BNPL credit model."
)


# ==========================================
# 1. PORTFOLIO OVERVIEW
# ==========================================

if page == "Portfolio Overview":

    st.header("Portfolio Overview")

    render_anchor_nav([
    ("Application inputs", "application-inputs"),
])

    risk_categories = ["All"] + sorted(
        customers["Risk_Segment"].dropna().unique().tolist()
    )

    selected_category = st.selectbox(
        "Risk category filter",
        risk_categories
    )

    if selected_category == "All":
        dashboard_customers = customers
    else:
        dashboard_customers = customers[
            customers["Risk_Segment"] == selected_category
        ]

    render_anchor("portfolio-key-metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", f"{len(dashboard_customers):,}")
    col2.metric(
        "Total EAD",
        f"{dashboard_customers['EAD'].sum():,.0f}"
    )
    col3.metric(
        "Average PD",
        f"{dashboard_customers['PD'].mean():.2%}"
    )
    col4.metric(
        "Default Rate",
        f"{dashboard_customers['actual_default'].mean():.2%}"
    )

    st.download_button(
        label="Download Excel report",
        data=create_excel_report(),
        file_name="bnpl_credit_risk_report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )

    st.divider()

    render_anchor("portfolio-risk-indicators")
    st.subheader("Portfolio Risk Indicators")

    col1, col2 = st.columns(2)

    col1.metric("Average Predicted PD", f"{average_pd:.2%}")
    col2.metric("Total Expected Credit Loss", f"{total_ecl:,.2f}")

    st.info(
        "Portfolio PD figures use the uncalibrated out-of-fold "
        "XGBoost predictions used in the ECL calculation."
    )

# ==========================================
# 2. CUSTOMER RISK
# ==========================================

elif page == "Customer Risk":

    st.header("Customer Risk Analysis")

    render_anchor_nav([
        ("Segment summary", "customer-risk-segment-summary"),
        ("Risk segment charts", "customer-risk-segment-charts"),
        ("Risk explorer", "customer-risk-explorer"),
        ("Customer profile", "customer-risk-profile"),
        ("Repayment history", "customer-risk-repayment-history"),
    ])

    st.write(
        "Explore customer risk categories, expected losses "
        "and individual customer profiles."
    )

    render_anchor("customer-risk-segment-summary")
    st.subheader("Risk Segment Summary")

    st.dataframe(segments, use_container_width=True)

    render_anchor("customer-risk-segment-charts")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        segments["Risk_Segment"].astype(str),
        segments["Customer_Count"]
    )

    ax.set_title("Customers by Risk Segment")
    ax.set_xlabel("Risk Segment")
    ax.set_ylabel("Number of Customers")

    plt.xticks(rotation=20)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        segments["Risk_Segment"].astype(str),
        segments["Total_ECL"]
    )

    ax.set_title("Expected Credit Loss by Risk Segment")
    ax.set_xlabel("Risk Segment")
    ax.set_ylabel("Total ECL")

    plt.xticks(rotation=20)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # CUSTOMER RISK EXPLORER

    st.divider()
    render_anchor("customer-risk-explorer")
    st.subheader("Customer Risk Explorer")

    risk_options = [
        "All",
        "Low Risk",
        "Medium Risk",
        "High Risk",
        "Very High Risk"
    ]

    selected_risk = st.selectbox(
        "Filter by risk category",
        risk_options
    )

    if selected_risk != "All":

        filtered_customers = customers[
            customers["Risk_Segment"] == selected_risk
        ]

    else:

        filtered_customers = customers.copy()

    st.metric(
        "Customers displayed",
        f"{len(filtered_customers):,}"
    )

    display_columns = [
        "PD", "LGD", "EAD", "ECL",
        "Risk_Segment", "actual_default"
    ]

    st.dataframe(
        filtered_customers[display_columns].sort_values(
            "PD", ascending=False
        ),
        use_container_width=True
    )

    st.download_button(
        label="Download filtered customer data",
        data=filtered_customers.to_csv(index=False),
        file_name="customer_risk_analysis.csv",
        mime="text/csv"
    )

    # INDIVIDUAL CUSTOMER PROFILE

    st.divider()
    render_anchor("customer-risk-profile")
    st.subheader("Individual Customer Risk Profile")

    customer_number = st.selectbox(
        "Select a customer",
        options=range(len(customers)),
        format_func=lambda x: f"Customer {x + 1}"
    )

    selected_customer = customers.iloc[customer_number]

    st.write(f"### Customer {customer_number + 1}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Predicted Default Probability",
        f"{selected_customer['PD']:.2%}"
    )

    col2.metric(
        "Risk Category",
        selected_customer["Risk_Segment"]
    )

    col3.metric(
        "Expected Credit Loss",
        f"{selected_customer['ECL']:,.2f}"
    )

    col4.metric(
        "Exposure at Default",
        f"{selected_customer['EAD']:,.2f}"
    )

    # REPAYMENT HISTORY

    st.divider()
    render_anchor("customer-risk-repayment-history")
    st.subheader("Historical Repayment Behaviour")

    customer_history = repayment.iloc[customer_number]

    repayment_months = pd.DataFrame({
        "Month": [
            "Month 1 (Most Recent)",
            "Month 2",
            "Month 3",
            "Month 4",
            "Month 5",
            "Month 6 (Oldest)"
        ],
        "Repayment Status": [
            customer_history["PAY_0"],
            customer_history["PAY_2"],
            customer_history["PAY_3"],
            customer_history["PAY_4"],
            customer_history["PAY_5"],
            customer_history["PAY_6"]
        ],
        "Payment Amount": [
            customer_history["PAY_AMT1"],
            customer_history["PAY_AMT2"],
            customer_history["PAY_AMT3"],
            customer_history["PAY_AMT4"],
            customer_history["PAY_AMT5"],
            customer_history["PAY_AMT6"]
        ]
    })

    st.dataframe(
        repayment_months,
        use_container_width=True
    )

    # CUSTOMER RISK DETAILS

    st.divider()
    st.subheader("Customer Risk Details")

    st.write({
        "Probability of Default": f"{selected_customer['PD']:.2%}",
        "Loss Given Default": f"{selected_customer['LGD']:.2%}",
        "Exposure at Default": f"{selected_customer['EAD']:,.2f}",
        "Expected Credit Loss": f"{selected_customer['ECL']:,.2f}",
        "Risk Segment": selected_customer["Risk_Segment"],
        "Actual Default": (
            "Yes" if selected_customer["actual_default"] == 1
            else "No"
        )
    })


# ==========================================
# 3. STRESS TESTING
# ==========================================

elif page == "Stress Testing":

    st.header("Credit Risk Stress Testing")

    render_anchor_nav([
        ("Scenario summary", "stress-scenario-summary"),
        ("ECL comparison", "stress-ecl-comparison"),
        ("Scenario notes", "stress-scenario-notes"),
    ])

    render_anchor("stress-scenario-summary")
    st.subheader("Scenario Summary")

    st.dataframe(
        stress,
        use_container_width=True
    )

    st.divider()

    render_anchor("stress-ecl-comparison")
    st.subheader("Expected Credit Loss Comparison")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        stress["Scenario"],
        stress["Total ECL"]
    )

    ax.set_title("Expected Credit Loss Under Stress Scenarios")
    ax.set_ylabel("Total ECL")

    plt.xticks(rotation=15)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    render_anchor("stress-scenario-notes")
    st.subheader("Scenario Notes")

    st.caption(
        "Stress scenarios are illustrative assumptions, "
        "not economic forecasts."
    )


# ==========================================
# 4. MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.header("Model Performance Comparison")

    render_anchor_nav([
        ("Evaluation results", "performance-evaluation-results"),
        ("Discrimination", "performance-discrimination"),
        ("Default recall", "performance-default-recall"),
    ])

    render_anchor("performance-evaluation-results")
    st.subheader("Model Evaluation Results")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.divider()

    render_anchor("performance-discrimination")
    st.subheader("Model Discrimination Performance")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        comparison["Model"],
        comparison["ROC_AUC"]
    )

    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Model Discrimination Performance")

    plt.xticks(rotation=15)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    render_anchor("performance-default-recall")
    st.subheader("Default Recall Comparison")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        comparison["Model"],
        comparison["Default_Recall"]
    )

    ax.set_ylabel("Default Recall")
    ax.set_ylim(0, 1)
    ax.set_title("Ability to Identify Actual Defaulters")

    plt.xticks(rotation=15)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 5. MODEL EXPLAINABILITY
# ==========================================

elif page == "Model Explainability":

    st.header("XGBoost Model Explainability")

    render_anchor_nav([
        ("SHAP summary plot", "explainability-shap-summary"),
        ("Feature importance", "explainability-feature-importance"),
    ])

    st.write(
        "SHAP helps explain which features influenced "
        "the model's predictions."
    )

    render_anchor("explainability-shap-summary")
    st.subheader("SHAP Summary Plot")

    st.image(
        "outputs/xgboost_shap_summary.png",
        use_container_width=True
    )

    st.divider()

    render_anchor("explainability-feature-importance")
    st.subheader("SHAP Feature Importance")

    st.image(
        "outputs/xgboost_shap_importance.png",
        use_container_width=True
    )


# ==========================================
# 6. CUSTOMER APPLICATION ASSESSMENT
# ==========================================

elif page == "Customer Application Assessment":

    st.header("BNPL Customer Application Assessment")

    render_anchor_nav([
        ("Application inputs", "application-inputs"),
    ])

    st.warning(
        "This is an educational simulation using historical "
        "credit-card data, not a validated BNPL lending model."
    )

    st.divider()

    render_anchor("application-inputs")
    st.subheader("Customer Application")

    col1, col2 = st.columns(2)

    with col1:

        customer_index = st.selectbox(
            "Select sample customer profile",
            options=list(range(min(100, len(engineered)))),
            format_func=lambda x: f"Sample customer {x + 1}"
        )

        purchase_amount = st.number_input(
            "Purchase amount (£)",
            min_value=10.0,
            max_value=5000.0,
            value=300.0,
            step=50.0
        )

    with col2:

        instalments = st.selectbox(
            "Number of instalments",
            options=[3, 4, 6, 12]
        )

        monthly_budget = st.number_input(
            "Available monthly budget (£)",
            min_value=0.0,
            max_value=10000.0,
            value=200.0,
            step=25.0
        )

    st.caption(
        "Available monthly budget means money remaining each "
        "month after essential expenses and existing commitments."
    )

    if st.button(
        "Assess application",
        type="primary"
    ):

        customer = engineered.iloc[customer_index]

        result = assess_application(
            customer=customer,
            purchase_amount=purchase_amount,
            instalments=instalments,
            available_monthly_budget=monthly_budget
        )

        st.divider()

        
        st.subheader("Assessment Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Default Probability",
            f"{result['default_probability']}%"
        )

        col2.metric(
            "Risk Category",
            result["risk_category"]
        )

        col3.metric(
            "Monthly Instalment",
            f"£{result['monthly_instalment']:.2f}"
        )

        st.divider()

        
        st.subheader("Affordability Assessment")

        col1, col2 = st.columns(2)

        col1.metric(
            "Available Monthly Budget",
            f"£{result['available_monthly_budget']:.2f}"
        )

        ratio = result["affordability_ratio"]

        col2.metric(
            "Budget Used by Instalment",
            f"{ratio:.1f}%" if ratio is not None else "N/A"
        )

        st.write(
            "**Affordability Status:**",
            result["affordability_status"]
        )

        st.divider()

       
        st.subheader("Decision Support Recommendation")

        st.markdown(
            f"### {result['recommendation']}"
        )

        st.caption(
            "Recommendation generated using illustrative "
            "prototype thresholds. It must not be used for "
            "actual lending."
        )

        with st.expander("View Full Assessment Details"):

            st.json(result)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "BNPL Credit Risk & Affordability Prediction System | "
    "Educational Analytics Prototype"
)