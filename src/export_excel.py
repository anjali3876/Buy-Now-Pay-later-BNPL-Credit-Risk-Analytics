
import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import BarChart, Reference
from openpyxl.utils import get_column_letter

# Project locations
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
EXCEL_FILE = ROOT / "BNPL_Credit_Risk_Analytics.xlsx"

# Read existing project results
portfolio = pd.read_csv(OUTPUTS / "portfolio_summary.csv")
models = pd.read_csv(OUTPUTS / "model_comparison.csv")
customer_risk = pd.read_csv(OUTPUTS / "customer_risk_segments.csv")
risk_summary = pd.read_csv(OUTPUTS / "risk_segmentation_summary.csv")
ecl = pd.read_csv(OUTPUTS / "customer_ecl_results.csv")
stress = pd.read_csv(OUTPUTS / "stress_testing_results.csv")
features = pd.read_csv(OUTPUTS / "feature_importance.csv")

# Create five-sheet Excel workbook
with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:

    # 1. Portfolio overview
    portfolio.to_excel(
        writer, sheet_name="Portfolio_Overview", index=False, startrow=2
    )
    models.to_excel(
        writer, sheet_name="Portfolio_Overview", index=False, startrow=12
    )

    # 2. Customer risk
    customer_risk.to_excel(
        writer, sheet_name="Customer_Risk", index=False
    )
    risk_summary.to_excel(
        writer, sheet_name="Customer_Risk", index=False, startrow=customer_risk.shape[0] + 4
    )

    # 3. ECL analysis
    ecl.to_excel(writer, sheet_name="ECL_Analysis", index=False)

    # 4. Stress testing
    stress.to_excel(writer, sheet_name="Stress_Testing", index=False)

    # 5. Executive dashboard data
    portfolio.to_excel(
        writer, sheet_name="Executive_Dashboard", index=False, startrow=2
    )
    risk_summary.to_excel(
        writer, sheet_name="Executive_Dashboard", index=False, startrow=12
    )
    stress.to_excel(
        writer, sheet_name="Executive_Dashboard", index=False, startrow=21
    )
    features.to_excel(
        writer, sheet_name="Executive_Dashboard", index=False, startrow=29
    )

# Apply professional formatting
wb = load_workbook(EXCEL_FILE)

navy = "17365D"
blue = "4472C4"
white = "FFFFFF"

for ws in wb.worksheets:
    ws.freeze_panes = "A2"
    ws.sheet_view.showGridLines = False

    for row in ws.iter_rows():
        for cell in row:
            if cell.value is not None:
                cell.alignment = Alignment(
                    vertical="center",
                    horizontal="left"
                )

    # Format column headings
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is not None and cell.row in (1, 3, 13, 14, 22, 23, 30):
                cell.fill = PatternFill(
                    "solid", fgColor=navy
                )
                cell.font = Font(
                    color=white, bold=True
                )

    # Adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)

        for cell in column:
            if cell.value is not None:
                max_length = max(
                    max_length, len(str(cell.value))
                )

        ws.column_dimensions[column_letter].width = min(
            max(max_length + 3, 14), 30
        )

# Add portfolio risk chart
ws = wb["Executive_Dashboard"]

chart = BarChart()
chart.title = "Customer Risk Segmentation"
chart.y_axis.title = "Number of Customers"
chart.x_axis.title = "Risk Segment"
chart.style = 10

# Risk summary is positioned at row 13
data = Reference(ws, min_col=2, min_row=13, max_row=17)
categories = Reference(ws, min_col=1, min_row=14, max_row=17)

chart.add_data(data, titles_from_data=True)
chart.set_categories(categories)
chart.height = 8
chart.width = 16

ws.add_chart(chart, "H2")

wb.save(EXCEL_FILE)

print("\nExcel workbook successfully created!")
print(f"Location: {EXCEL_FILE}")
print("Worksheets: 5")
print("Source: Existing BNPL project outputs")
