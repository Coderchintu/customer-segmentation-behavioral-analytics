"""Exploratory Data Analysis: plots of sales, products, countries and customers.

Each function returns a matplotlib Figure and, when `fig_dir` is given,
also saves it as a PNG file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from src.rfm_analysis import RFM_FEATURES
from src.utils import CURRENCY, save_figure, set_plot_style

set_plot_style()
money_fmt = FuncFormatter(lambda x, _: f"{CURRENCY}{x:,.0f}")


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue, orders and active customers per calendar month."""
    monthly = (
        df.assign(Month=df["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month")
        .agg(Revenue=("TotalAmount", "sum"),
             Orders=("Invoice", "nunique"),
             Customers=("CustomerID", "nunique"))
        .reset_index()
    )
    # Flag the final month if the data stops before the month ends.
    last_day = df["InvoiceDate"].max()
    monthly["PartialMonth"] = False
    if last_day.day < last_day.days_in_month:
        monthly.loc[monthly.index[-1], "PartialMonth"] = True
    return monthly


def _mark_partial_month(ax, monthly: pd.DataFrame, column: str) -> None:
    partial = monthly[monthly["PartialMonth"]]
    if not partial.empty:
        ax.scatter(partial["Month"], partial[column], s=80, facecolors="none",
                   edgecolors="red", zorder=5, label="Partial month")
        ax.legend(loc="upper left")


def plot_monthly_revenue(df: pd.DataFrame, fig_dir: Path | None = None) -> plt.Figure:
    monthly = monthly_summary(df)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["Month"], monthly["Revenue"], marker="o")
    _mark_partial_month(ax, monthly, "Revenue")
    ax.yaxis.set_major_formatter(money_fmt)
    ax.set(title="Monthly Revenue", xlabel="Month", ylabel="Revenue")
    fig.autofmt_xdate()
    save_figure(fig, "01_monthly_revenue.png", fig_dir)
    return fig


def plot_monthly_orders(df: pd.DataFrame, fig_dir: Path | None = None) -> plt.Figure:
    monthly = monthly_summary(df)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly["Month"], monthly["Orders"], width=20, color="tab:blue", alpha=0.8)
    ax.set(title="Number of Orders per Month", xlabel="Month", ylabel="Orders")
    fig.autofmt_xdate()
    save_figure(fig, "02_monthly_orders.png", fig_dir)
    return fig


def plot_top_products(df: pd.DataFrame, n: int = 10, fig_dir: Path | None = None) -> plt.Figure:
    top = (df.groupby("Description")["TotalAmount"].sum()
           .nlargest(n).sort_values())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top.index.str.title(), top.values, color="tab:green")
    ax.xaxis.set_major_formatter(money_fmt)
    ax.set(title=f"Top {n} Products by Revenue", xlabel="Revenue", ylabel="")
    save_figure(fig, "03_top_products.png", fig_dir)
    return fig


def plot_top_countries(df: pd.DataFrame, n: int = 10, fig_dir: Path | None = None) -> plt.Figure:
    revenue = df.groupby("Country")["TotalAmount"].sum().sort_values(ascending=False)
    share = revenue / revenue.sum() * 100
    top = revenue.head(n).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top.index, top.values, color="tab:purple")
    ax.set_xscale("log")  # one country usually dominates; log scale keeps others visible
    for i, (country, value) in enumerate(top.items()):
        ax.text(value, i, f"  {share[country]:.1f}%", va="center", fontsize=8)
    ax.set(title=f"Top {n} Countries by Revenue (log scale, % of total)",
           xlabel="Revenue (log scale)", ylabel="")
    save_figure(fig, "04_top_countries.png", fig_dir)
    return fig


def plot_order_value_distribution(df: pd.DataFrame, fig_dir: Path | None = None) -> plt.Figure:
    order_values = df.groupby("Invoice")["TotalAmount"].sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(order_values, bins=60, log_scale=True, ax=ax, color="tab:orange")
    ax.axvline(order_values.median(), color="black", linestyle="--",
               label=f"Median = {CURRENCY}{order_values.median():,.0f}")
    ax.legend()
    ax.set(title="Revenue Distribution per Order (log scale)",
           xlabel="Order value (log scale)", ylabel="Number of orders")
    save_figure(fig, "05_order_value_distribution.png", fig_dir)
    return fig


def plot_customer_frequency(rfm: pd.DataFrame, fig_dir: Path | None = None) -> plt.Figure:
    capped = rfm["Frequency"].clip(upper=30)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(capped, bins=np.arange(0.5, 31.5, 1), color="tab:cyan", edgecolor="white")
    ax.set(title="Customer Purchase Frequency (orders per customer, capped at 30)",
           xlabel="Number of orders", ylabel="Number of customers")
    save_figure(fig, "06_customer_frequency.png", fig_dir)
    return fig


def plot_customer_monetary(rfm: pd.DataFrame, fig_dir: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(rfm["Monetary"], bins=60, log_scale=True, ax=ax, color="tab:red")
    ax.axvline(rfm["Monetary"].median(), color="black", linestyle="--",
               label=f"Median = {CURRENCY}{rfm['Monetary'].median():,.0f}")
    ax.legend()
    ax.set(title="Customer Monetary Value (total spend, log scale)",
           xlabel="Total spend (log scale)", ylabel="Number of customers")
    save_figure(fig, "07_customer_monetary.png", fig_dir)
    return fig


def plot_rfm_distributions(rfm: pd.DataFrame, fig_dir: Path | None = None) -> plt.Figure:
    """Raw RFM distributions (top row) vs log-transformed (bottom row)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, feature in enumerate(RFM_FEATURES):
        sns.histplot(rfm[feature], bins=50, ax=axes[0, i], color=f"C{i}")
        axes[0, i].set_title(f"{feature} (raw), skew = {rfm[feature].skew():.1f}")
        logged = np.log1p(rfm[feature])
        sns.histplot(logged, bins=50, ax=axes[1, i], color=f"C{i}")
        axes[1, i].set_title(f"log(1 + {feature}), skew = {logged.skew():.1f}")
    fig.suptitle("Distribution of RFM Variables: Raw vs Log-Transformed", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "08_rfm_distributions.png", fig_dir)
    return fig


def plot_correlation_heatmap(rfm: pd.DataFrame, fig_dir: Path | None = None) -> plt.Figure:
    columns = RFM_FEATURES + ["AvgOrderValue"]
    corr = rfm[columns].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, ax=ax)
    ax.set_title("Spearman Correlation of Customer Features")
    save_figure(fig, "09_correlation_heatmap.png", fig_dir)
    return fig


def run_eda(df: pd.DataFrame, rfm: pd.DataFrame, fig_dir: Path | None = None) -> None:
    """Create and save every EDA figure."""
    plots = [
        lambda: plot_monthly_revenue(df, fig_dir),
        lambda: plot_monthly_orders(df, fig_dir),
        lambda: plot_top_products(df, fig_dir=fig_dir),
        lambda: plot_top_countries(df, fig_dir=fig_dir),
        lambda: plot_order_value_distribution(df, fig_dir),
        lambda: plot_customer_frequency(rfm, fig_dir),
        lambda: plot_customer_monetary(rfm, fig_dir),
        lambda: plot_rfm_distributions(rfm, fig_dir),
        lambda: plot_correlation_heatmap(rfm, fig_dir),
    ]
    for make_plot in plots:
        plt.close(make_plot())
