# Data

## Source

This project uses the **Online Retail** dataset from the UCI Machine Learning Repository.

- Page: https://archive.ics.uci.edu/dataset/352/online+retail
- Citation: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

It contains 541,909 transactions from a UK-based online retailer between 01/12/2010 and 09/12/2011. Amounts are in pounds sterling (£).

| Column | Description |
|---|---|
| InvoiceNo | Invoice number. Starts with `C` for cancellations |
| StockCode | Product code |
| Description | Product name |
| Quantity | Units per transaction line |
| InvoiceDate | Date and time of the invoice |
| UnitPrice | Price per unit (£) |
| CustomerID | Customer identifier (missing for about 25% of rows) |
| Country | Customer's country |

The code renames `InvoiceNo` to `Invoice`. It also accepts other common spellings (`Price`, `Customer ID`), so datasets with the same structure, such as *Online Retail II*, can be used too.

## Folders

| Folder | Contents | In Git? |
|---|---|---|
| `raw/` | Full dataset as `online_retail.csv` (~48 MB) | No, downloaded |
| `processed/` | SQLite database created by the pipeline | No, generated |
| `sample/` | Small sample for quick runs (~1.7 MB) | Yes |

## Getting the full dataset

**Automatic** (recommended):

```bash
python -m src.download_data
```

This downloads the zip from UCI, converts the Excel file to CSV, and saves it as `data/raw/online_retail.csv`. `python run_pipeline.py` also does this automatically when the file is missing.

**Manual** (if the download is blocked on your network):

1. Download the zip from the UCI page above and extract it.
2. Copy `Online Retail.xlsx` into `data/raw/`.
3. Run `python -m src.download_data` to convert it to CSV.

## Sample dataset

`sample/online_retail_sample.csv` has 19,208 rows. It contains all transactions of **200 randomly selected customers** (NumPy seed 42) plus **800 random rows without a CustomerID**. It keeps real cancellations, duplicates and missing IDs, so every cleaning step is exercised. It is redistributed under the dataset's CC BY 4.0 license with the attribution above.

Results from the sample are for testing only. Reported results in the main README come from the full dataset.
