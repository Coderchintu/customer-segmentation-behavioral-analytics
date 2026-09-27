"""Shared paths, constants and small helper functions used across the project."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, so the project runs on any machine)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SQL_DIR = PROJECT_ROOT / "sql"

RAW_CSV = RAW_DIR / "online_retail.csv"
SAMPLE_CSV = SAMPLE_DIR / "online_retail_sample.csv"
DB_PATH = PROCESSED_DIR / "retail_analytics.db"
SQL_FILE = SQL_DIR / "analytics_queries.sql"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
CURRENCY = "£"  # the UCI Online Retail dataset is recorded in pounds sterling

# The project works with these standard column names.
STANDARD_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]

# Different versions of retail datasets use slightly different column names.
# Any alias on the left is renamed to the standard name on the right.
COLUMN_ALIASES = {
    "InvoiceNo": "Invoice",
    "Invoice No": "Invoice",
    "Price": "UnitPrice",
    "Unit Price": "UnitPrice",
    "Customer ID": "CustomerID",
    "Customer_ID": "CustomerID",
    "Invoice Date": "InvoiceDate",
}


def get_logger(name: str) -> logging.Logger:
    """Return a simple console logger."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    return logging.getLogger(name)


def ensure_dirs(*dirs: Path) -> None:
    """Create directories if they do not exist."""
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    """Apply one consistent, readable plotting style."""
    sns.set_theme(style="whitegrid", palette="deep", font_scale=0.95)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["axes.titleweight"] = "bold"


def save_figure(fig: plt.Figure, filename: str, fig_dir: Path | None) -> Path | None:
    """Save a figure as PNG inside fig_dir. Returns the path (or None if fig_dir is None)."""
    if fig_dir is None:
        return None
    ensure_dirs(fig_dir)
    path = Path(fig_dir) / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def rel(path: Path) -> str:
    """Show a path relative to the project root (keeps logs free of machine-specific paths)."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def format_currency(value: float) -> str:
    """Format a number as currency, e.g. 1234.5 -> '£1,234.50'."""
    return f"{CURRENCY}{value:,.2f}"
