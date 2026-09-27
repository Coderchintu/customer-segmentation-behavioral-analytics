"""Basic statistical analysis of customer-level RFM data.

Named `statistical_analysis.py` (not `statistics.py`) so it never shadows
Python's built-in `statistics` module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.rfm_analysis import RFM_FEATURES


def descriptive_stats(rfm: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Mean, median, standard deviation, quartiles, IQR, skewness and kurtosis."""
    columns = columns or RFM_FEATURES
    data = rfm[columns]
    summary = pd.DataFrame({
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "min": data.min(),
        "Q1": data.quantile(0.25),
        "Q3": data.quantile(0.75),
        "max": data.max(),
    })
    summary["IQR"] = summary["Q3"] - summary["Q1"]
    summary["skewness"] = data.skew()
    summary["kurtosis"] = data.kurt()
    return summary.round(2)


def iqr_outlier_counts(rfm: pd.DataFrame, columns: list[str] | None = None) -> pd.Series:
    """Count values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] for each column."""
    columns = columns or RFM_FEATURES
    counts = {}
    for col in columns:
        q1, q3 = rfm[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        counts[col] = int(((rfm[col] < q1 - 1.5 * iqr) | (rfm[col] > q3 + 1.5 * iqr)).sum())
    return pd.Series(counts, name="iqr_outliers")


def skewness_before_after_log(rfm: pd.DataFrame) -> pd.DataFrame:
    """Compare skewness of raw vs log1p-transformed RFM features."""
    raw = rfm[RFM_FEATURES].skew()
    logged = np.log1p(rfm[RFM_FEATURES]).skew()
    return pd.DataFrame({"raw_skewness": raw, "log_skewness": logged}).round(2)


def correlation_matrix(rfm: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """Correlation between RFM features.

    Spearman (rank) correlation is the default because it is robust to the
    heavy skew and outliers typical of spending data.
    """
    return rfm[RFM_FEATURES].corr(method=method).round(3)


def compare_domestic_vs_international(
    rfm: pd.DataFrame, domestic_country: str = "United Kingdom", column: str = "AvgOrderValue"
) -> dict:
    """Mann-Whitney U test: do domestic and international customers differ in `column`?

    H0: the distributions of `column` are the same for both groups.
    H1: the distributions differ (two-sided).
    Mann-Whitney is used instead of a t-test because spending is heavily
    skewed and not normally distributed.
    """
    domestic = rfm.loc[rfm["Country"] == domestic_country, column]
    international = rfm.loc[rfm["Country"] != domestic_country, column]
    if len(domestic) < 5 or len(international) < 5:
        return {"test": "Mann-Whitney U", "note": "Not enough customers in one of the groups."}

    u_stat, p_value = stats.mannwhitneyu(domestic, international, alternative="two-sided")
    # Rank-biserial correlation: effect size in [-1, 1]. 0 = no difference,
    # positive = group A tends to have larger values, negative = smaller.
    effect_size = (2 * u_stat) / (len(domestic) * len(international)) - 1
    return {
        "test": "Mann-Whitney U (two-sided)",
        "variable": column,
        "group_a": domestic_country,
        "group_b": "International",
        "n_group_a": int(len(domestic)),
        "n_group_b": int(len(international)),
        "median_group_a": round(float(domestic.median()), 2),
        "median_group_b": round(float(international.median()), 2),
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "rank_biserial_effect_size": round(float(effect_size), 3),
        "significant_at_0.05": bool(p_value < 0.05),
    }
