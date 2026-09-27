"""Download the UCI Online Retail dataset and store it as a CSV file.

Source : UCI Machine Learning Repository, "Online Retail" (dataset id 352)
         https://archive.ics.uci.edu/dataset/352/online+retail
License: Creative Commons Attribution 4.0 (CC BY 4.0)

Usage (from the project root):
    python -m src.download_data          # download only if missing
    python -m src.download_data --force  # download again
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import RAW_CSV, RAW_DIR, ensure_dirs, get_logger, rel

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
EXCEL_NAME = "Online Retail.xlsx"

logger = get_logger(__name__)

MANUAL_INSTRUCTIONS = f"""
Automatic download failed. Please download the dataset manually:
  1. Open https://archive.ics.uci.edu/dataset/352/online+retail
  2. Click "Download" and extract the zip file.
  3. Copy "{EXCEL_NAME}" into the folder: data/raw/
  4. Run:  python -m src.download_data
     (the Excel file will be converted to {RAW_CSV.name})
"""


def excel_to_csv(excel_path: Path, csv_path: Path = RAW_CSV) -> Path:
    """Convert the original Excel file to CSV (CSV loads roughly 20x faster)."""
    logger.info("Reading %s (this can take ~30 seconds)...", excel_path.name)
    df = pd.read_excel(excel_path, dtype={"InvoiceNo": str, "StockCode": str})
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s rows to %s", f"{len(df):,}", rel(csv_path))
    return csv_path


def download_dataset(force: bool = False) -> Path:
    """Make sure data/raw/online_retail.csv exists and return its path."""
    ensure_dirs(RAW_DIR)

    if RAW_CSV.exists() and not force:
        logger.info("Dataset already available: %s", rel(RAW_CSV))
        return RAW_CSV

    # If the user already placed the Excel file in data/raw, just convert it.
    local_excel = RAW_DIR / EXCEL_NAME
    if local_excel.exists() and not force:
        return excel_to_csv(local_excel)

    zip_path = RAW_DIR / "online_retail.zip"
    try:
        logger.info("Downloading dataset from UCI (~23 MB)...")
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extract(EXCEL_NAME, RAW_DIR)
    except (urllib.error.URLError, zipfile.BadZipFile, KeyError, OSError) as error:
        logger.error("Download failed: %s", error)
        raise SystemExit(MANUAL_INSTRUCTIONS) from error
    finally:
        zip_path.unlink(missing_ok=True)

    csv_path = excel_to_csv(local_excel)
    local_excel.unlink(missing_ok=True)  # keep only the CSV to save space
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the UCI Online Retail dataset.")
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args()
    download_dataset(force=args.force)
