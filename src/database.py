"""Create the SQLite database and run the analytics queries in sql/analytics_queries.sql.

Usage (from the project root, after running the pipeline):
    python -m src.database            # print every query result
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from src.utils import DB_PATH, SQL_FILE, ensure_dirs, get_logger, rel

logger = get_logger(__name__)


def create_database(tables: dict[str, pd.DataFrame], db_path: Path = DB_PATH) -> Path:
    """Write each DataFrame to a table (replacing old versions) and add indexes."""
    ensure_dirs(Path(db_path).parent)
    with sqlite3.connect(db_path) as conn:
        for name, frame in tables.items():
            frame = frame.copy()
            # Store datetimes as ISO text so SQLite's strftime() works on them.
            for col in frame.select_dtypes(include=["datetime", "datetimetz"]).columns:
                frame[col] = frame[col].dt.strftime("%Y-%m-%d %H:%M:%S")
            frame.to_sql(name, conn, if_exists="replace", index=False)

        if "transactions" in tables:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_customer ON transactions(CustomerID)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_invoice ON transactions(Invoice)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(InvoiceDate)")
    logger.info("SQLite database written: %s (tables: %s)", rel(db_path), ", ".join(tables))
    return Path(db_path)


def load_queries(sql_file: Path = SQL_FILE) -> dict[str, str]:
    """Read named queries. Each query starts with a line like '-- name: total_revenue'."""
    text = Path(sql_file).read_text(encoding="utf-8")
    parts = re.split(r"^--\s*name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    # parts = [header, name1, body1, name2, body2, ...]
    queries = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        queries[name] = body.strip()
    if not queries:
        raise ValueError(f"No named queries found in {sql_file}")
    return queries


def run_query(sql: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Run one SQL query and return the result as a DataFrame."""
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Database not found at {rel(db_path)}. Run `python run_pipeline.py` first."
        )
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def run_all_queries(db_path: Path = DB_PATH, sql_file: Path = SQL_FILE,
                    output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Run every named query; optionally save each result as CSV."""
    results = {}
    for name, sql in load_queries(sql_file).items():
        results[name] = run_query(sql, db_path)
        if output_dir is not None:
            ensure_dirs(output_dir)
            results[name].to_csv(Path(output_dir) / f"{name}.csv", index=False)
    logger.info("Ran %d SQL queries.", len(results))
    return results


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    for query_name, result in run_all_queries().items():
        print(f"\n=== {query_name} ===")
        print(result.to_string(index=False))
