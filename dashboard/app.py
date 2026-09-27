"""Streamlit dashboard for customer segmentation and behavioral analytics.

Run from the project root (after `python run_pipeline.py`):
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.database import load_queries, run_query  # noqa: E402
from src.utils import DB_PATH, SQL_FILE  # noqa: E402

st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid", font_scale=0.9)

PAGES = [
    "Overview",
    "Sales & Revenue",
    "Customer Analysis",
    "RFM Analysis",
    "Customer Segmentation",
    "Cluster Visualization",
    "Segment Summary",
    "SQL Queries",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data(db_path: str, modified_time: float) -> dict:
    """Read all tables once. `modified_time` refreshes the cache when the DB changes."""
    with sqlite3.connect(db_path) as conn:
        # Load only the columns the dashboard needs.
        transactions = pd.read_sql(
            "SELECT Invoice, Description, InvoiceDate, TotalAmount, CustomerID, Country "
            "FROM transactions", conn, parse_dates=["InvoiceDate"])
        customers = pd.read_sql("SELECT * FROM customers", conn)
        profile = pd.read_sql("SELECT * FROM segment_profile", conn)
        k_eval = pd.read_sql("SELECT * FROM k_evaluation", conn)
        meta_rows = pd.read_sql("SELECT * FROM metadata", conn)
    # Repeated text values stored as categories use far less memory.
    for col in ["Invoice", "Description", "CustomerID", "Country"]:
        transactions[col] = transactions[col].astype("category")
    metadata = {row.key: json.loads(row.value) for row in meta_rows.itertuples()}
    return {"transactions": transactions, "customers": customers,
            "profile": profile, "k_eval": k_eval, "metadata": metadata}


if not DB_PATH.exists():
    st.title("Customer Segmentation Dashboard")
    st.error(
        "No processed data found. From the project root, run one of:\n\n"
        "`python run_pipeline.py --sample`  (quick, bundled sample)\n\n"
        "`python run_pipeline.py`  (full UCI dataset)\n\n"
        "then reload this page."
    )
    st.stop()

data = load_data(str(DB_PATH), DB_PATH.stat().st_mtime)
tx_all, cust_all = data["transactions"], data["customers"]
profile, k_eval, meta = data["profile"], data["k_eval"], data["metadata"]
CUR = meta.get("currency", "£")


def money(value: float) -> str:
    return f"{CUR}{value:,.0f}"


def short_name(segment: str) -> str:
    """'High-Value Customers' -> 'High-Value' so chart labels are not cut off."""
    return segment.replace(" Customers", "")


# ---------------------------------------------------------------------------
# Sidebar: navigation and filters
# ---------------------------------------------------------------------------
st.sidebar.title("Customer Analytics")
page = st.sidebar.radio("Section", PAGES)

st.sidebar.header("Filters")
countries = st.sidebar.multiselect(
    "Country", sorted(tx_all["Country"].unique().tolist()), placeholder="All countries"
)
segments = st.sidebar.multiselect(
    "Customer segment", sorted(cust_all["Segment"].unique()), placeholder="All segments"
)
min_date = tx_all["InvoiceDate"].min().date()
max_date = tx_all["InvoiceDate"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
st.sidebar.caption(
    "The date range filters transactions. RFM values and segments are always "
    f"computed on the full period (snapshot date {meta['snapshot_date']})."
)

# Apply filters
customers = cust_all.copy()
if countries:
    customers = customers[customers["Country"].isin(countries)]
if segments:
    customers = customers[customers["Segment"].isin(segments)]

transactions = tx_all[tx_all["CustomerID"].isin(customers["CustomerID"])]
if countries:
    transactions = transactions[transactions["Country"].isin(countries)]
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    transactions = transactions[(transactions["InvoiceDate"] >= start)
                                & (transactions["InvoiceDate"] < end)]

if transactions.empty or customers.empty:
    st.warning("No data matches the selected filters. Clear a filter in the sidebar to continue.")
    st.stop()


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    return (df.assign(Month=df["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
              .groupby("Month")
              .agg(Revenue=("TotalAmount", "sum"), Orders=("Invoice", "nunique"))
              .reset_index())


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
st.title(page)
st.caption(f"Data: {meta['data_source']}  |  Customers shown: {len(customers):,}")

if page == "Overview":
    revenue = transactions["TotalAmount"].sum()
    orders = transactions["Invoice"].nunique()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total revenue", money(revenue))
    c2.metric("Customers", f"{transactions['CustomerID'].nunique():,}")
    c3.metric("Orders", f"{orders:,}")
    c4.metric("Average order value", f"{CUR}{revenue / orders:,.2f}")

    st.subheader("Monthly revenue")
    st.line_chart(monthly(transactions), x="Month", y="Revenue")
    if transactions["InvoiceDate"].max().day < 28:
        st.caption("The last month may be partial, so its lower value is not a real drop.")

    left, right = st.columns(2)
    by_segment = (customers.assign(Segment=customers["Segment"].map(short_name))
                  .groupby("Segment")
                  .agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"))
                  .reset_index())
    left.subheader("Customers per segment")
    left.bar_chart(by_segment, x="Segment", y="Customers", horizontal=True,
                   sort="-Customers", x_label="")
    right.subheader("Revenue per segment")
    right.bar_chart(by_segment, x="Segment", y="Revenue", horizontal=True,
                    sort="-Revenue", x_label="")

elif page == "Sales & Revenue":
    m = monthly(transactions)
    left, right = st.columns(2)
    left.subheader("Revenue trend")
    left.line_chart(m, x="Month", y="Revenue")
    right.subheader("Orders per month")
    right.bar_chart(m, x="Month", y="Orders")

    left, right = st.columns(2)
    top_products = (transactions.groupby("Description", observed=True)["TotalAmount"].sum()
                    .nlargest(10).rename("Revenue").reset_index())
    left.subheader("Top 10 products by revenue")
    left.bar_chart(top_products, x="Description", y="Revenue", horizontal=True, sort="-Revenue")
    by_country = (transactions.groupby("Country", observed=True)["TotalAmount"].sum()
                  .nlargest(10).rename("Revenue").reset_index())
    right.subheader("Top 10 countries by revenue")
    right.bar_chart(by_country, x="Country", y="Revenue", horizontal=True, sort="-Revenue")

elif page == "Customer Analysis":
    left, right = st.columns(2)
    by_country = customers["Country"].value_counts().head(10).rename_axis("Country").reset_index(name="Customers")
    left.subheader("Customers by country (top 10)")
    left.bar_chart(by_country, x="Country", y="Customers", horizontal=True, sort="-Customers")

    freq = customers["Frequency"].clip(upper=20).value_counts().sort_index()
    freq_df = pd.DataFrame({"Orders (20 = 20 or more)": freq.index, "Customers": freq.values})
    right.subheader("Orders per customer")
    right.bar_chart(freq_df, x="Orders (20 = 20 or more)", y="Customers")

    st.subheader("Total spend per customer (log scale)")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.histplot(customers["Monetary"], bins=50, log_scale=True, ax=ax, color="#4C72B0")
    ax.set(xlabel=f"Total spend ({CUR}, log scale)", ylabel="Customers")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Top 10 customers by spend")
    top = customers.nlargest(10, "Monetary")[
        ["CustomerID", "Country", "Frequency", "Monetary", "Recency", "Segment"]]
    st.dataframe(top, hide_index=True)

elif page == "RFM Analysis":
    st.markdown(
        "**Recency**: days since last purchase (lower is better). "
        "**Frequency**: number of orders. **Monetary**: total spend."
    )
    stats = customers[["Recency", "Frequency", "Monetary"]].describe().T
    stats["median"] = customers[["Recency", "Frequency", "Monetary"]].median()
    stats["skewness"] = customers[["Recency", "Frequency", "Monetary"]].skew()
    st.dataframe(stats.round(2))

    left, right = st.columns([2, 1])
    feature = left.selectbox("Variable", ["Recency", "Frequency", "Monetary"])
    use_log = left.checkbox("Apply log(1 + x) transform", value=feature != "Recency")
    values = np.log1p(customers[feature]) if use_log else customers[feature]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.histplot(values, bins=50, ax=ax, color="#55A868")
    ax.set(xlabel=f"log(1 + {feature})" if use_log else feature, ylabel="Customers",
           title=f"Skewness = {values.skew():.2f}")
    left.pyplot(fig)
    plt.close(fig)

    corr = customers[["Recency", "Frequency", "Monetary"]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman correlation")
    right.pyplot(fig)
    plt.close(fig)

    st.subheader("RFM score distribution (sum of 1-5 quintile scores)")
    score_counts = customers["RFM_Score"].value_counts().sort_index()
    st.bar_chart(pd.DataFrame({"RFM score": score_counts.index, "Customers": score_counts.values}),
                 x="RFM score", y="Customers")

elif page == "Customer Segmentation":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters (k)", meta["selected_k"])
    c2.metric("Silhouette score", f"{meta['silhouette']:.3f}")
    c3.metric("Davies-Bouldin", f"{meta['davies_bouldin']:.3f}")
    c4.metric("Stability (ARI)", f"{meta['stability_ari']:.3f}")
    st.markdown(
        f"Features were log-transformed and standardised before K-Means. The elbow "
        f"method suggests **k = {meta['elbow_k']}**; the rule used here picks the highest "
        f"silhouette score for k ≥ 3, giving **k = {meta['selected_k']}**. Stability is the "
        "average agreement (Adjusted Rand Index) between runs with different random seeds."
    )
    left, right = st.columns(2)
    left.subheader("Elbow method (inertia)")
    left.line_chart(k_eval, x="k", y="inertia")
    right.subheader("Silhouette score")
    right.line_chart(k_eval, x="k", y="silhouette")

    st.subheader("Customers per cluster")
    sizes = (customers.groupby(["Cluster", "Segment"]).size()
             .reset_index(name="Customers"))
    sizes["Cluster"] = "Cluster " + sizes["Cluster"].astype(str) + ": " + sizes["Segment"]
    st.bar_chart(sizes, x="Cluster", y="Customers", horizontal=True, x_label="")

elif page == "Cluster Visualization":
    variance = meta["pca_explained_variance"]
    st.scatter_chart(customers, x="PC1", y="PC2", color="Segment", size=12)
    st.caption(
        f"PC1 explains {variance[0] * 100:.1f}% and PC2 {variance[1] * 100:.1f}% of the variance "
        "in the scaled RFM features. PCA is used only to draw the clusters in 2D; it does not "
        "prove the clusters are 'correct'. K-Means works in the full 3-feature space, so some "
        "overlap between colours in this 2D view would be normal."
    )

elif page == "Segment Summary":
    st.info(
        "Segment names are rule-based interpretations of each cluster's average RFM values, "
        "not ground-truth customer categories."
    )
    summary = (customers.groupby("Segment")
               .agg(**{"Customers": ("CustomerID", "count"),
                       "Median recency (days)": ("Recency", "median"),
                       "Median orders": ("Frequency", "median"),
                       f"Median spend ({CUR})": ("Monetary", "median"),
                       f"Revenue ({CUR})": ("Monetary", "sum")})
               .sort_values(f"Revenue ({CUR})", ascending=False))
    summary["Revenue share (%)"] = summary[f"Revenue ({CUR})"] / summary[f"Revenue ({CUR})"].sum() * 100
    st.dataframe(summary.round(1).reset_index(), hide_index=True,
                 column_config={"Segment": st.column_config.TextColumn(width="medium")})

    st.subheader("Segment-level RFM comparison (medians)")
    chart_data = summary.reset_index()
    chart_data["Segment"] = chart_data["Segment"].map(short_name)
    cols = st.columns(3)
    features = ["Median recency (days)", "Median orders", f"Median spend ({CUR})"]
    for col, feature in zip(cols, features):
        col.bar_chart(chart_data, x="Segment", y=feature, horizontal=True, x_label="")

    st.subheader("Interpretation and suggested actions")
    for row in profile.sort_values("Revenue", ascending=False).itertuples():
        if segments and row.Segment not in segments:
            continue
        with st.expander(f"Cluster {row.Cluster}: {row.Segment}"):
            st.write(row.Description)
            st.write(f"**Suggested action:** {row.SuggestedAction}")

elif page == "SQL Queries":
    st.markdown("Predefined queries from `sql/analytics_queries.sql`, run on the SQLite "
                "database. Sidebar filters do not apply here.")
    queries = load_queries(SQL_FILE)
    name = st.selectbox("Query", list(queries), format_func=lambda n: n.replace("_", " ").title())
    st.code(queries[name], language="sql")
    try:
        st.dataframe(run_query(queries[name], DB_PATH), hide_index=True)
    except Exception as error:  # show the SQL error instead of crashing the app
        st.error(f"Query failed: {error}")
