import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import zipfile
from io import BytesIO

# ============================================
# PAGE TITLE
# ============================================
st.set_page_config(page_title="E-Commerce BI Dashboard", layout="wide")
st.title("📊 E-Commerce BI Dashboard")

# ============================================
# DATA LOADING FROM GITHUB ZIP
# ============================================
@st.cache_data
def load_data_from_github():
    url = "https://github.com/rushiraj-gohil/data-cleanup-and-analysis/raw/refs/heads/main/cleaned_data.zip"

    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Failed to download dataset from GitHub.")
        st.stop()

    zip_file = zipfile.ZipFile(BytesIO(response.content))

    transactions = pd.read_csv(zip_file.open("cleaned_transactions.csv"), parse_dates=["created_at"])
    sessions = pd.read_csv(zip_file.open("cleaned_sessions.csv"), parse_dates=["session_start", "session_end"])
    customers = pd.read_csv(zip_file.open("cleaned_customers.csv"), parse_dates=["signup_date"])
    tickets = pd.read_csv(zip_file.open("cleaned_support_tickets.csv"), parse_dates=["created_at", "resolved_at"])
    products = pd.read_csv(zip_file.open("cleaned_products.csv"))

    return transactions, sessions, customers, tickets, products


transactions, sessions, customers, tickets, products = load_data_from_github()


# ============================================
# SECTION 1: REVENUE TREND + ANOMALY DETECTION
# ============================================
st.header("1️⃣ Monthly Revenue Trend with Anomaly Detection")

# Filter PAID transactions
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

# Revenue trend chart
rev_chart = (
    alt.Chart(monthly_rev)
    .mark_line(point=True)
    .encode(
        x=alt.X("transaction_month:T", title="Month"),
        y=alt.Y("total_amount:Q", title="Revenue"),
        color=alt.condition(
            alt.datum.anomaly == "Anomaly",
            alt.value("red"),
            alt.value("#1f77b4")
        ),
        tooltip=["transaction_month", "total_amount", "anomaly"]
    )
    .properties(height=350)
)

st.altair_chart(rev_chart, use_container_width=True)
st.info("🔍 **Insight:** Red points indicate revenue anomalies using a 2+ Z-score deviation.")


# ============================================
# SECTION 2: COHORT RETENTION HEATMAP
# ============================================
st.header("2️⃣ Cohort Retention Heatmap (0–5 Months)")

# Cohort preparation
customers["cohort_month"] = customers["signup_date"].dt.to_period("M").dt.to_timestamp()
sessions["activity_month"] = sessions["session_start"].dt.to_period("M").dt.to_timestamp()

merge = pd.merge(
    customers[["customer_id", "cohort_month"]],
    sessions[["customer_id", "activity_month"]],
    on="customer_id",
    how="left"
)

merge["month_number"] = (
    (merge["activity_month"].dt.year - merge["cohort_month"].dt.year) * 12 +
    (merge["activity_month"].dt.month - merge["cohort_month"].dt.month)
)

merge = merge[(merge["month_number"] >= 0) & (merge["month_number"] <= 5)]

cohort_size = merge.groupby("cohort_month")["customer_id"].nunique()

retention = (
    merge.groupby(["cohort_month", "month_number"])["customer_id"]
    .nunique()
    .unstack(fill_value=0)
)

retention_rate = retention.divide(cohort_size, axis=0).round(3) * 100

st.dataframe(
    retention_rate.style.background_gradient(cmap="Blues"),
    use_container_width=True
)

st.info("📘 **Insight:** Heatmap shows how well each cohort retains users in months 0–5.")


# ============================================
# SECTION 3: SUPPORT TICKETS VS PAYMENT STATUS
# ============================================
st.header("3️⃣ Support Ticket Volume vs Payment Outcomes")

# Aggregate support tickets
ticket_counts = tickets.groupby("customer_id").size().reset_index(name="ticket_count")

# Aggregate transactions
payment_summary = (
    transactions.groupby(["customer_id", "payment_status"])["transaction_id"]
    .count()
    .unstack(fill_value=0)
    .reset_index()
)

support_vs_payment = pd.merge(ticket_counts, payment_summary, on="customer_id", how="left")
support_vs_payment.fillna(0, inplace=True)

support_vs_payment["paid_tx"] = support_vs_payment.get("paid", 0)

# Scatter plot
scatter_plot = (
    alt.Chart(support_vs_payment)
    .mark_circle(size=80)
    .encode(
        x=alt.X("ticket_count:Q", title="Support Tickets Raised"),
        y=alt.Y("paid_tx:Q", title="Paid Transactions"),
        color=alt.Color("charged_back:Q", scale=alt.Scale(scheme="redyellowblue")),
        tooltip=["customer_id", "ticket_count", "paid_tx", "refunded", "charged_back"]
    )
    .properties(height=350)
)

st.altair_chart(scatter_plot, use_container_width=True)

st.info("💬 **Insight:** Customers with more support tickets often have higher refund/chargeback behavior.")


# ============================================
# END
# ============================================
st.success("🎉 Dashboard Loaded Successfully!")
