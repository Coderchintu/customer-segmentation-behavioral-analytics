"""Run the complete customer segmentation pipeline.

    python run_pipeline.py            # full UCI dataset (downloaded automatically)
    python run_pipeline.py --sample   # small bundled sample, runs in seconds
    python run_pipeline.py --k 4      # override the automatically selected k

Outputs
    data/processed/retail_analytics.db   SQLite database used by the dashboard
    models/*.joblib                      fitted scaler, K-Means and PCA
    reports/ (or reports/sample_run/)    figures, CSV tables and summary.json
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # save figures without opening windows

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import clustering, eda  # noqa: E402
from src import statistical_analysis as sa  # noqa: E402
from src.data_cleaning import clean_transactions, load_raw_data, validate_transactions  # noqa: E402
from src.database import create_database, run_all_queries  # noqa: E402
from src.download_data import download_dataset  # noqa: E402
from src.rfm_analysis import add_rfm_scores, calculate_rfm  # noqa: E402
from src.utils import (  # noqa: E402
    CURRENCY, DB_PATH, REPORTS_DIR, SAMPLE_CSV, ensure_dirs, get_logger,
)

logger = get_logger("pipeline")


def run(use_sample: bool = False, k: int | None = None) -> dict:
    report_dir = REPORTS_DIR / "sample_run" if use_sample else REPORTS_DIR
    fig_dir = report_dir / "figures"
    ensure_dirs(report_dir, fig_dir)

    # 1. Load --------------------------------------------------------------
    data_path = SAMPLE_CSV if use_sample else download_dataset()
    raw = load_raw_data(data_path)

    # 2-3. Clean and validate ----------------------------------------------
    transactions, cleaning_log = clean_transactions(raw)
    checks = validate_transactions(transactions)
    cleaning_log.to_csv(report_dir / "cleaning_log.csv", index=False)

    # 4. RFM ---------------------------------------------------------------
    rfm = add_rfm_scores(calculate_rfm(transactions))
    snapshot = transactions["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    # 5. Statistics --------------------------------------------------------
    desc = sa.descriptive_stats(rfm, ["Recency", "Frequency", "Monetary", "AvgOrderValue"])
    desc.to_csv(report_dir / "descriptive_stats.csv")
    skew = sa.skewness_before_after_log(rfm)
    outliers = sa.iqr_outlier_counts(rfm)
    correlation = sa.correlation_matrix(rfm)
    hypothesis = sa.compare_domestic_vs_international(rfm)

    # 6. EDA figures -------------------------------------------------------
    eda.run_eda(transactions, rfm, fig_dir)

    # 7. Scaling + choosing k ----------------------------------------------
    X, scaler = clustering.prepare_features(rfm)
    k_max = min(10, len(rfm) - 1)
    k_eval = clustering.evaluate_k_range(X, k_min=2, k_max=k_max)
    k_eval.to_csv(report_dir / "k_evaluation.csv", index=False)
    elbow_k = clustering.find_elbow(k_eval["k"], k_eval["inertia"])
    selected_k = k if k is not None else clustering.select_k(k_eval, min_k=3)
    logger.info("Elbow suggests k=%d; silhouette (k>=3) selects k=%d; using k=%d",
                elbow_k, clustering.select_k(k_eval, 3), selected_k)

    # 8. Final model -------------------------------------------------------
    model = clustering.train_kmeans(X, selected_k)
    labels = model.labels_
    stability = clustering.stability_check(X, selected_k)
    selected_row = k_eval[k_eval["k"] == selected_k].iloc[0]

    # 9. PCA (visualisation only) ------------------------------------------
    coords, pca = clustering.apply_pca(X)

    # 10. Interpretation ---------------------------------------------------
    profile = clustering.interpret_clusters(rfm, X, labels)
    profile.to_csv(report_dir / "segment_profile.csv", index=False)
    segment_map = profile.set_index("Cluster")["Segment"]

    customers = rfm.assign(
        Cluster=labels,
        Segment=pd.Series(labels).map(segment_map).values,
        PC1=coords[:, 0].round(4),
        PC2=coords[:, 1].round(4),
    )
    customers.to_csv(report_dir / "customer_segments.csv", index=False)

    clustering.plot_k_evaluation(k_eval, selected_k, elbow_k, fig_dir)
    clustering.plot_pca_clusters(coords, customers["Segment"], pca.explained_variance_ratio_, fig_dir)
    clustering.plot_segment_comparison(profile, fig_dir)
    plt.close("all")
    clustering.save_models(model, scaler, pca)

    # 11. SQLite + SQL analytics ------------------------------------------
    metadata = {
        "data_source": "sample" if use_sample else "UCI Online Retail (full)",
        "snapshot_date": snapshot.strftime("%Y-%m-%d"),
        "selected_k": selected_k,
        "elbow_k": elbow_k,
        "silhouette": round(float(selected_row["silhouette"]), 4),
        "davies_bouldin": round(float(selected_row["davies_bouldin"]), 4),
        "stability_ari": round(stability, 4),
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "currency": CURRENCY,
    }
    create_database({
        "transactions": transactions,
        "customers": customers,
        "segment_profile": profile,
        "k_evaluation": k_eval,
        "metadata": pd.DataFrame({"key": list(metadata), "value": [json.dumps(v) for v in metadata.values()]}),
    }, DB_PATH)
    sql_results = run_all_queries(DB_PATH, output_dir=report_dir / "sql_results")

    # 12. Summary ----------------------------------------------------------
    order_values = transactions.groupby("Invoice")["TotalAmount"].sum()
    summary = {
        **metadata,
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(transactions)),
        "date_range": [transactions["InvoiceDate"].min().strftime("%Y-%m-%d"),
                       transactions["InvoiceDate"].max().strftime("%Y-%m-%d")],
        "customers": int(transactions["CustomerID"].nunique()),
        "orders": int(transactions["Invoice"].nunique()),
        "countries": int(transactions["Country"].nunique()),
        "total_revenue": round(float(transactions["TotalAmount"].sum()), 2),
        "average_order_value": round(float(order_values.mean()), 2),
        "validation_checks": checks,
        "skewness": skew.to_dict(),
        "iqr_outliers": outliers.to_dict(),
        "spearman_correlation": correlation.to_dict(),
        "hypothesis_test": hypothesis,
        "segments": profile[["Cluster", "Segment", "Customers", "CustomerSharePct",
                             "RevenueSharePct", "Recency_median", "Frequency_median",
                             "Monetary_median"]].to_dict(orient="records"),
    }
    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print_summary(summary, sql_results)
    return summary


def print_summary(summary: dict, sql_results: dict) -> None:
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Data source      : {summary['data_source']}")
    print(f"Rows (raw/clean) : {summary['raw_rows']:,} / {summary['clean_rows']:,}")
    print(f"Customers        : {summary['customers']:,}   Orders: {summary['orders']:,}")
    print(f"Total revenue    : {CURRENCY}{summary['total_revenue']:,.2f}")
    print(f"Avg order value  : {CURRENCY}{summary['average_order_value']:,.2f}")
    print(f"Selected k       : {summary['selected_k']} (elbow suggests {summary['elbow_k']})")
    print(f"Silhouette       : {summary['silhouette']:.3f}   "
          f"Davies-Bouldin: {summary['davies_bouldin']:.3f}   "
          f"Stability ARI: {summary['stability_ari']:.3f}")
    print("\nSegments:")
    for seg in summary["segments"]:
        print(f"  - {seg['Segment']:<32} {seg['Customers']:>5} customers "
              f"({seg['CustomerSharePct']:.1f}%), {seg['RevenueSharePct']:.1f}% of revenue")
    print(f"\nSQL queries run  : {len(sql_results)}")
    print("Next step        : streamlit run dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer segmentation pipeline")
    parser.add_argument("--sample", action="store_true", help="use the bundled sample dataset")
    parser.add_argument("--k", type=int, default=None, help="force a specific number of clusters")
    args = parser.parse_args()
    if args.k is not None and args.k < 2:
        parser.error("--k must be at least 2")
    run(use_sample=args.sample, k=args.k)
