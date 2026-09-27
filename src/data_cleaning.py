"""Data loading, cleaning and validation for transactional retail data.

Every cleaning step is logged with the number of rows it removed and the
reason, so no data is deleted silently. The log is saved to
reports/cleaning_log.csv by the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import COLUMN_ALIASES, STANDARD_COLUMNS, get_logger

logger = get_logger(__name__)

# Stock codes that are services or accounting entries, not products.
# POST/DOT/C2 = postage and carriage, D = discount, M = manual adjustment,
# S = samples, B = bad-debt adjustment, CRUK = charity commission,
# PADS = padding charge, BANK CHARGES / AMAZONFEE = fees.
NON_PRODUCT_CODES = {
    "POST", "DOT", "C2", "D", "M", "S", "B",
    "CRUK", "PADS", "BANK CHARGES", "AMAZONFEE",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column aliases to standard names and check required columns."""
    df = df.rename(columns={c: COLUMN_ALIASES.get(c.strip(), c.strip()) for c in df.columns})
    missing = [col for col in STANDARD_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df[STANDARD_COLUMNS].copy()


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel transaction file and return it with standard columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path.name}\n"
            "Run `python -m src.download_data` or use the sample: "
            "`python run_pipeline.py --sample`."
        )

    # Invoice and StockCode look numeric but contain letters (e.g. 'C536379', '85123A').
    text_columns = {"InvoiceNo": str, "Invoice": str, "StockCode": str}
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=text_columns)
    else:
        df = pd.read_csv(path, dtype=text_columns, encoding="utf-8")

    df = standardize_columns(df)
    logger.info("Loaded %s rows from %s", f"{len(df):,}", path.name)
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def _remove_matched_cancellations(
    purchases: pd.DataFrame, cancellations: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Remove purchase rows that were later fully cancelled.

    A cancellation is matched to a purchase when CustomerID, StockCode,
    UnitPrice and absolute Quantity are identical. Matching is one-to-one:
    each cancellation removes at most one purchase row. This is a simple,
    explainable rule; partial returns are not matched.
    """
    keys = ["CustomerID", "StockCode", "Quantity", "UnitPrice"]

    cancel_keys = cancellations[keys].copy()
    cancel_keys["Quantity"] = cancel_keys["Quantity"].abs()
    cancel_keys["UnitPrice"] = cancel_keys["UnitPrice"].round(4)
    cancel_keys["match_no"] = cancel_keys.groupby(keys).cumcount()

    purchase_keys = purchases[keys].copy()
    purchase_keys["UnitPrice"] = purchase_keys["UnitPrice"].round(4)
    purchase_keys["match_no"] = purchase_keys.groupby(keys).cumcount()
    purchase_keys["row_id"] = purchases.index

    matched = purchase_keys.merge(cancel_keys, on=keys + ["match_no"], how="inner")
    return purchases.drop(index=matched["row_id"]), len(matched)


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw transactions.

    Returns
    -------
    clean_df : cleaned purchase-level data with a TotalAmount column
    log_df   : one row per cleaning step (rows removed, rows remaining, reason)
    """
    df = df.copy().reset_index(drop=True)
    log = []

    def record(step: str, before: int, reason: str) -> None:
        log.append({
            "step": step,
            "rows_removed": before - len(df),
            "rows_remaining": len(df),
            "reason": reason,
        })

    log.append({"step": "Raw data", "rows_removed": 0, "rows_remaining": len(df),
                "reason": "Starting point"})

    # 1. Exact duplicates: identical rows are almost certainly double-logged entries.
    before = len(df)
    df = df.drop_duplicates()
    record("Remove exact duplicates", before, "Identical rows are repeated records")

    # 2. Dates: convert to datetime; unparseable dates cannot be used in time analysis.
    before = len(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    record("Parse InvoiceDate", before, "Rows with invalid dates cannot be placed in time")

    # 3. Customer ID: RFM needs to know WHO bought; anonymous rows cannot be attributed.
    before = len(df)
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype("int64").astype(str)
    record("Remove missing CustomerID", before,
           "Transactions without a customer cannot be used for customer-level analysis")

    # Normalise text columns
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Description"] = df["Description"].fillna("UNKNOWN").astype(str).str.strip()

    # 4. Cancellations: invoices starting with 'C'. Remove the purchases they cancel,
    #    then drop the cancellation rows themselves (they are not purchases).
    before = len(df)
    is_cancel = df["Invoice"].str.startswith("C")
    cancellations = df[is_cancel]
    purchases = df[~is_cancel]
    purchases, n_matched = _remove_matched_cancellations(purchases, cancellations)
    df = purchases
    record("Handle cancellations", before,
           f"{int(is_cancel.sum()):,} cancellation rows removed; {n_matched:,} original "
           "purchase rows they fully cancelled were also removed")

    # 5. Non-positive quantities left after removing cancellations are adjustments.
    before = len(df)
    df = df[df["Quantity"] > 0]
    record("Remove non-positive Quantity", before,
           "Zero/negative quantities outside cancellations are stock adjustments")

    # 6. Prices must be positive for a real sale (0 = free items/adjustments).
    before = len(df)
    df = df[df["UnitPrice"] > 0]
    record("Remove non-positive UnitPrice", before,
           "Zero or negative prices are not genuine sales")

    # 7. Service / accounting stock codes (postage, fees, discounts, manual entries).
    before = len(df)
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]
    record("Remove non-product codes", before,
           "Postage, fees, discounts and manual entries do not describe product purchases")

    # 8. Feature: revenue per line
    df = df.copy()
    df["TotalAmount"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df = df.sort_values("InvoiceDate").reset_index(drop=True)

    log_df = pd.DataFrame(log)
    logger.info("Cleaning finished: %s rows kept out of %s",
                f"{len(df):,}", f"{log[0]['rows_remaining']:,}")
    return df, log_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_transactions(df: pd.DataFrame) -> dict[str, bool]:
    """Run simple sanity checks on cleaned data. Raises ValueError if any check fails."""
    checks = {
        "no_missing_key_values": bool(
            df[["Invoice", "StockCode", "InvoiceDate", "CustomerID"]].notna().all().all()
        ),
        "quantity_positive": bool((df["Quantity"] > 0).all()),
        "unit_price_positive": bool((df["UnitPrice"] > 0).all()),
        "total_amount_positive": bool((df["TotalAmount"] > 0).all()),
        "no_cancellation_invoices": bool(~df["Invoice"].str.startswith("C").any()),
        "invoice_date_is_datetime": pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]),
        "no_duplicate_rows": bool(~df.duplicated().any()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Data validation failed: {failed}")
    logger.info("All %d validation checks passed.", len(checks))
    return checks
