"""Unit tests on small hand-made data, so expected results can be checked by hand.

Run from the project root:  python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

from src import clustering
from src.data_cleaning import clean_transactions, standardize_columns, validate_transactions
from src.database import create_database, load_queries, run_all_queries
from src.rfm_analysis import add_rfm_scores, calculate_rfm


def make_row(invoice, code, qty, date, price, customer, country="United Kingdom"):
    return {"InvoiceNo": invoice, "StockCode": code, "Description": f"Item {code}",
            "Quantity": qty, "InvoiceDate": date, "UnitPrice": price,
            "CustomerID": customer, "Country": country}


@pytest.fixture
def raw_transactions():
    rows = [
        make_row("1001", "A1", 2, "2011-01-01 10:00", 5.0, 111.0),
        make_row("1001", "A1", 2, "2011-01-01 10:00", 5.0, 111.0),     # exact duplicate
        make_row("1002", "B2", 1, "2011-01-10 12:00", 20.0, 111.0),
        make_row("1003", "A1", 10, "2011-01-05 09:00", 5.0, 222.0),    # cancelled below
        make_row("C1004", "A1", -10, "2011-01-06 09:00", 5.0, 222.0),  # cancellation
        make_row("1005", "B2", 3, "2011-01-20 15:00", 20.0, 222.0),
        make_row("1006", "A1", 1, "2011-01-15 11:00", 5.0, np.nan),    # no customer
        make_row("1007", "POST", 1, "2011-01-15 11:00", 18.0, 111.0),  # postage
        make_row("1008", "B2", 1, "2011-01-16 11:00", 0.0, 111.0),     # zero price
        make_row("1009", "C3", 4, "2011-01-31 08:00", 2.5, 333.0, "France"),
    ]
    return standardize_columns(pd.DataFrame(rows))


def test_standardize_columns_renames_aliases():
    df = pd.DataFrame(columns=["Invoice", "StockCode", "Description", "Quantity",
                               "InvoiceDate", "Price", "Customer ID", "Country"])
    assert "UnitPrice" in standardize_columns(df).columns


def test_standardize_columns_reports_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        standardize_columns(pd.DataFrame(columns=["InvoiceNo", "Quantity"]))


def test_cleaning_removes_invalid_rows(raw_transactions):
    clean, log = clean_transactions(raw_transactions)
    # Kept: 1001 (once), 1002, 1005, 1009
    assert sorted(clean["Invoice"]) == ["1001", "1002", "1005", "1009"]
    assert log["rows_remaining"].iloc[-1] == 4
    assert (clean["TotalAmount"] == clean["Quantity"] * clean["UnitPrice"]).all()


def test_cancellation_removes_the_original_purchase(raw_transactions):
    clean, _ = clean_transactions(raw_transactions)
    assert "1003" not in set(clean["Invoice"])   # fully cancelled purchase
    assert not clean["Invoice"].str.startswith("C").any()


def test_validation_passes_on_clean_data(raw_transactions):
    clean, _ = clean_transactions(raw_transactions)
    assert all(validate_transactions(clean).values())


def test_rfm_values(raw_transactions):
    clean, _ = clean_transactions(raw_transactions)
    rfm = calculate_rfm(clean).set_index("CustomerID")
    # Snapshot = last date (2011-01-31) + 1 day = 2011-02-01
    assert rfm.loc["111", "Recency"] == 22        # last purchase 2011-01-10
    assert rfm.loc["111", "Frequency"] == 2
    assert rfm.loc["111", "Monetary"] == pytest.approx(30.0)
    assert rfm.loc["222", "Frequency"] == 1       # cancelled order not counted
    assert rfm.loc["333", "Recency"] == 1
    assert rfm.loc["333", "Country"] == "France"


def test_rfm_scores_are_between_1_and_5():
    rng = np.random.default_rng(0)
    rfm = pd.DataFrame({"CustomerID": range(50), "Recency": rng.integers(1, 300, 50),
                        "Frequency": rng.integers(1, 20, 50),
                        "Monetary": rng.uniform(10, 5000, 50)})
    scored = add_rfm_scores(rfm)
    for col in ["R_Score", "F_Score", "M_Score"]:
        assert scored[col].between(1, 5).all()


def synthetic_rfm(n_per_group=60):
    """Three clearly different customer groups."""
    rng = np.random.default_rng(42)
    groups = [(10, 15, 5000), (60, 3, 800), (300, 1, 100)]  # recency, frequency, monetary
    frames = []
    for i, (r, f, m) in enumerate(groups):
        frames.append(pd.DataFrame({
            "Recency": np.clip(rng.normal(r, r * 0.2, n_per_group), 1, None).round(),
            "Frequency": np.clip(rng.normal(f, f * 0.2, n_per_group), 1, None).round(),
            "Monetary": np.clip(rng.normal(m, m * 0.2, n_per_group), 1, None),
        }))
    rfm = pd.concat(frames, ignore_index=True)
    rfm.insert(0, "CustomerID", [str(i) for i in range(len(rfm))])
    return rfm


def test_clustering_finds_and_names_distinct_groups():
    rfm = synthetic_rfm()
    X, _ = clustering.prepare_features(rfm)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-9)

    k_eval = clustering.evaluate_k_range(X, 2, 6)
    assert clustering.select_k(k_eval, min_k=3) == 3

    model = clustering.train_kmeans(X, 3)
    profile = clustering.interpret_clusters(rfm, X, model.labels_)
    best = profile.loc[profile["Monetary_median"].idxmax(), "Segment"]
    worst = profile.loc[profile["Monetary_median"].idxmin(), "Segment"]
    assert best == "High-Value Customers"
    assert worst.startswith("Low-Engagement Customers")
    assert profile["Segment"].is_unique  # duplicate names get a "(Cluster n)" suffix
    assert clustering.stability_check(X, 3) > 0.9


def test_find_elbow():
    k = [2, 3, 4, 5, 6, 7, 8]
    inertia = [1000, 400, 200, 170, 150, 140, 135]
    assert clustering.find_elbow(k, inertia) == 4


def test_sql_queries_run(raw_transactions, tmp_path):
    clean, _ = clean_transactions(raw_transactions)
    customers = calculate_rfm(clean).assign(Segment="Test segment")
    db_path = create_database({"transactions": clean, "customers": customers},
                              tmp_path / "test.db")
    queries = load_queries()
    assert len(queries) >= 8

    results = run_all_queries(db_path)
    assert results["total_revenue"].iloc[0, 0] == pytest.approx(clean["TotalAmount"].sum())
    assert results["number_of_orders"]["total_orders"].iloc[0] == 4
    assert results["average_order_value"]["average_order_value"].iloc[0] == pytest.approx(
        clean.groupby("Invoice")["TotalAmount"].sum().mean(), abs=0.01)
