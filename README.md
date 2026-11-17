
# Buy Now, Pay Later (BNPL) Credit Risk Analytics

An educational Streamlit dashboard for exploring credit risk analytics in a Buy Now, Pay Later context. It brings portfolio measures, customer risk profiles, stress testing, model comparison, SHAP explainability, and a sample application affordability assessment into one interface.

> **Important:** This is an educational prototype, not a validated BNPL model or a real lending decision system. It uses historical credit-card data as a proxy.

## Features

- **Portfolio overview:** customer count, exposure at default (EAD), average predicted probability of default (PD), default rate, and expected credit loss (ECL).
- **Customer risk analysis:** risk categories, risk-segment summaries, customer profiles, repayment history, filters, and a filtered customer CSV download.
- **Stress testing:** illustrative scenarios and their expected credit loss results.
- **Model performance:** model comparison, including ROC-AUC and default recall.
- **Model explainability:** SHAP summary and feature-importance visualizations for the XGBoost model.
- **Application assessment:** a sample customer profile, purchase amount, instalment count, and monthly budget are used to display an illustrative risk and affordability assessment.
- **Excel report:** download portfolio, risk-segment, stress-test, model-performance, customer-risk, and ECL summary sheets.

## Technology

- Python
- Streamlit
- pandas
- Matplotlib
- scikit-learn
- XGBoost
- SHAP
- openpyxl

See [`requirements.txt`](requirements.txt) for the project dependencies.

## Project structure

```text
.
├── data/
│   ├── default of credit card clients.xls
│   ├── cleaned_credit_data.csv
│   └── engineered_credit_data.csv
├── database/
│   └── bnpl_risk.db
├── outputs/
│   ├── customer_ecl_results.csv
│   ├── risk_segmentation_summary.csv
│   ├── stress_testing_results.csv
│   ├── portfolio_summary.csv
│   ├── customer_risk_segments.csv
│   ├── model_comparison.csv
│   ├── xgboost_shap_summary.png
│   └── xgboost_shap_importance.png
└── src/
    ├── dashboard.py
    ├── decision_engine.py
    ├── clean_data.py
    ├── feature_engineering.py
    ├── train_xgboost.py
    ├── train_random_forest.py
    ├── calculate_ecl.py
    ├── stress_testing.py
    ├── model_comparison.py
    └── ...
```

The dashboard reads files from `data/` and `outputs/` using paths relative to the project root. Keep the required files in those folders when running it.

## Run the dashboard

From the project’s top-level directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/dashboard.py
```

To stop the dashboard, press **Control+C** in Terminal.

## Data source

This project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository. The data describes historical credit-card clients in Taiwan and includes demographic, repayment, bill, and payment information. It is not BNPL transaction data.

**Citation**

Yeh, I-C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

The UCI dataset page lists the dataset under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. See the [UCI dataset page](https://archive.ics.uci.edu/dataset/350/default%2Bcredit%2Bcard%2Bclients) for its description and license. The project includes processed versions and derived model outputs.

## Limitations

- Historical credit-card data may not represent BNPL customers, products, or repayment behaviour.
- The model and prototype thresholds have not been validated for real lending decisions.
- Model performance metrics describe results on this project’s data and evaluation setup; they do not establish suitability for production.
- SHAP visualizations explain model predictions; they do not establish that the predictions are correct or fair.
- The application assessment is illustrative and should not be used to make real credit decisions.

## Responsible use
Educational prototype using historical credit-card data. Results are illustrative and have not been validated for real BNPL lending decisions.
