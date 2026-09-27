"""RFM (Recency, Frequency, Monetary) analysis.

Recency   = days between the customer's last purchase and the snapshot date
Frequency = number of distinct orders (invoices)
Monetary  = total amount spent

The snapshot date is one day after the last transaction in the data, so the
most recent customers get Recency = 1 rather than 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RFM_FEATURES = ["Recency", "Frequency", "Monetary"]


def calculate_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build one row per customer with Recency, Frequency and Monetary values.

    Extra descriptive columns (not used for clustering):
    Country (the customer's most common country), AvgOrderValue,
    FirstPurchase and LastPurchase.
    """
    if df.empty:
        raise ValueError("Cannot calculate RFM on an empty DataFrame.")

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        FirstPurchase=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalAmount", "sum"),
    )
    # Compare calendar dates (not times) so recency is a whole number of days.
    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"].dt.normalize()).dt.days
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]

    # A few customers ordered from more than one country; keep the most common one.
    country = df.groupby("CustomerID")["Country"].agg(lambda s: s.mode().iloc[0])
    rfm["Country"] = country

    rfm = rfm.reset_index()
    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["AvgOrderValue"] = rfm["AvgOrderValue"].round(2)
    columns = ["CustomerID", "Recency", "Frequency", "Monetary",
               "AvgOrderValue", "Country", "FirstPurchase", "LastPurchase"]
    return rfm[columns]


def add_rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add classic 1-5 quintile scores (5 = best) as a simple rule-based baseline.

    Frequency has many ties (lots of customers with 1 order), so values are
    ranked first so that qcut can always create five equal-sized groups.
    """
    rfm = rfm.copy()
    labels = [1, 2, 3, 4, 5]
    # Lower recency is better, so its labels are reversed.
    rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=labels[::-1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=labels).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=labels).astype(int)
    rfm["RFM_Score"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    return rfm


def log_transform(rfm: pd.DataFrame) -> pd.DataFrame:
    """Return log(1 + x) of the RFM features to reduce right skew."""
    return np.log1p(rfm[RFM_FEATURES])
