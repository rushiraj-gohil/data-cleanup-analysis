import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import zipfile
from io import BytesIO

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="E-Commerce BI Dashboard", layout="wide")
st.title("📊 E-Commerce BI Dashboard")
st.write("This dashboard fulfills the BI tasks from the Data Analyst assignment.")
st.write("---")

# =====================================================
# LOAD CLEANED DATASETS FROM GITHUB ZIP
# =====================================================
@st.cache_data
def load_data():
    url = "https://github.com/rushiraj-gohil/data-cleanup-and-analysis/raw/refs/heads/main/cleaned_data.zip"

    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Failed to download cleaned_data.zip")
        st.stop()

    zip_file = zipfile.ZipFile(BytesIO(response.content))

    transactions = pd.read_csv(zip_file.open("cleaned_transactions.csv"), parse_dates=["created_at"])
    sessions = pd.read_csv(zip_file.open("cleaned_sessions.csv"), parse_dates=["session_start", "session_end"])
    customers = pd.read_csv(zip_file.open("cleaned_customers.csv"), parse_dates=["signup_date"])
    tickets = pd.read_csv(zip_file.open("cleaned_support_tickets.csv"), parse_dates=["created_at", "resolved_at"])
    products = pd.read_csv(zip_file.open("cleaned_products.csv"))

    return transactions, sessions, customers, tickets, products


transactions, sessions, customers, tickets, products = load_data()


# =====================================================
# 1️⃣ REVENUE TREND + ANOMALY DETECTION
# =====================================================
st.header("1️⃣ Revenue Trend with Anomaly Detection")

paid_tx = transactions[transactions["payment_status"] == "paid"].copy()
paid_tx["transaction_month"] = paid_tx["created_at"].dt.to_period("M").dt.to_timestamp()

monthly_rev = (
    paid_tx.groupby("transaction_month")["total_amount"]
    .sum()
    .reset_index()
    .sort_values("transaction_month")
)

# Z-score anomalies
mean_rev = monthly_rev["total_amount"].mean()
std_rev = monthly_rev["total_amount"].std()

monthly_rev["z_score"] = (monthly_rev["total_amount"] - mean_rev) / std_rev
monthly_rev["anomaly"] = np.where(abs(monthly_rev["z_score"]) > 2, "Anomaly", "Normal")

line_chart = (
    alt.Chart(monthly_rev)
    .mark_line(point=True)
    .encode(
        x="transaction_month:T",
        y="total_amount:Q",
        color=alt.condition(
            alt.datum.anomaly == "Anomaly",
            alt.value("red"),
            alt.value("#1f77b4")
        ),
        tooltip=["transaction_month", "total_amount", "anomaly"]
    )
    .properties(height=350)
)

st.altair_chart(line_chart, use_container_width=True)

st.info("""
### Business Question Answered  
**How does revenue change over time, and which months deviate significantly?**

This helps PMs detect:
- Seasonality  
- Spikes due to promotions  
- Drops due to outages or friction  
- Unusual anomalies requiring investigation  
""")


# =====================================================
# 2️⃣ COHORT RETENTION VISUALIZATION
# =====================================================
st.header("2️⃣ Cohort Retention Visualization (0–5 months)")

customers["cohort_month"] = customers["signup_date"].dt.to_period("M").dt.to_timestamp()
sessions["activity_month"] = sessions["session_start"].dt.to_period("M").dt.to_timestamp()

merged = pd.merge(
    customers[["customer_id", "cohort_month"]],
    sessions[["customer_id", "activity_month"]],
    on="customer_id",
    how="left"
)

merged["month_number"] = (
    (merged["activity_month"].dt.year - merged["cohort_month"].dt.year) * 12 +
    (merged["activity_month"].dt.month - merged["cohort_month"].dt.month)
)

merged = merged[(merged["month_number"] >= 0) & (merged["month_number"] <= 5)]

cohort_size = merged.groupby("cohort_month")["customer_id"].nunique()

retention = (
    merged.groupby(["cohort_month", "month_number"])["customer_id"]
    .nunique()
    .unstack(fill_value=0)
)

retention_rate = retention.divide(cohort_size, axis=0).round(3) * 100

# Display retention table
st.subheader("Cohort Retention Table (0–5 months)")
st.dataframe(retention_rate, use_container_width=True)

st.info("""
### Business Question Answered  
**How well do customer cohorts retain over their first six months?**

This reveals:
- True retention behavior  
- Onboarding quality  
- Month-to-month drop-offs  
- Which cohorts performed best and why  
""")


# =====================================================
# 3️⃣ SUPPORT TICKETS VS PAYMENT STATUS
# =====================================================
st.header("3️⃣ Support Ticket Volume vs Payment Status")

ticket_counts = tickets.groupby("customer_id").size().reset_index(name="ticket_count")

payment_summary = (
    transactions.groupby(["customer_id", "payment_status"])["transaction_id"]
    .count()
    .unstack(fill_value=0)
    .reset_index()
)

combined = pd.merge(ticket_counts, payment_summary, on="customer_id", how="left")
combined.fillna(0, inplace=True)

combined["paid_tx"] = combined.get("paid", 0)

scatter_plot = (
    alt.Chart(combined)
    .mark_circle(size=75)
    .encode(
        x=alt.X("ticket_count:Q", title="Support Tickets Raised"),
        y=alt.Y("paid_tx:Q", title="Paid Transactions"),
        color=alt.Color("charged_back:Q", scale=alt.Scale(scheme="redyellowblue")),
        tooltip=["customer_id", "ticket_count", "paid_tx", "refunded", "charged_back"]
    )
    .properties(height=350)
)

st.altair_chart(scatter_plot, use_container_width=True)

st.info("""
### Business Question Answered  
**Does customer support friction correlate with payment problems?**

Insights:
- Users with more tickets often have more refunds/chargebacks  
- Indicates product friction → revenue loss  
- Helps PM prioritize UX / support improvements  
""")

