
import streamlit as st
import pandas as pd

from decision_engine import assess_application

# Page configuration
st.set_page_config(
    page_title="BNPL Credit Risk Assessment",
    page_icon="💳",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/engineered_credit_data.csv")


df = load_data()

# Dashboard heading
st.title("BNPL Credit Risk & Affordability System")

st.caption(
    "Machine-learning powered credit risk assessment "
    "and simulated affordability analysis"
)

st.info(
    "Educational prototype using historical public credit data. "
    "Not a real lending decision or validated BNPL credit model."
)

st.divider()

# Application inputs
st.subheader("Customer application")

col1, col2 = st.columns(2)

with col1:
    customer_index = st.selectbox(
        "Select sample customer profile",
        options=list(range(min(100, len(df)))),
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
        options=[3, 4, 6, 12],
        index=0
    )

    monthly_budget = st.number_input(
        "Available monthly budget (£)",
        min_value=0.0,
        max_value=10000.0,
        value=200.0,
        step=25.0
    )

# Run assessment
if st.button("Assess application", type="primary"):

    customer = df.iloc[customer_index]

    result = assess_application(
        customer=customer,
        purchase_amount=purchase_amount,
        instalments=instalments,
        available_monthly_budget=monthly_budget
    )

    st.divider()
    st.subheader("Assessment results")

    # Main results
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted default probability",
        f"{result['default_probability']}%"
    )

    col2.metric(
        "Risk category",
        result["risk_category"]
    )

    col3.metric(
        "Monthly instalment",
        f"£{result['monthly_instalment']:.2f}"
    )

    # Affordability
    st.subheader("Affordability assessment")

    col1, col2 = st.columns(2)

    col1.metric(
        "Available monthly budget",
        f"£{result['available_monthly_budget']:.2f}"
    )

    ratio = result["affordability_ratio"]

    col2.metric(
        "Budget used by instalment",
        f"{ratio:.1f}%" if ratio is not None else "N/A"
    )

    st.write(
        "**Affordability status:**",
        result["affordability_status"]
    )

    # Recommendation
    st.subheader("Decision support recommendation")

    st.markdown(f"### {result['recommendation']}")

    st.caption(
        "Recommendation generated using illustrative prototype "
        "thresholds. It must not be used for actual lending."
    )

    # Detailed results
    with st.expander("View full assessment details"):
        st.json(result)
