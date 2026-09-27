"""Customer segmentation with K-Means, cluster evaluation, PCA and interpretation."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.rfm_analysis import RFM_FEATURES
from src.utils import CURRENCY, MODELS_DIR, RANDOM_STATE, ensure_dirs, save_figure, set_plot_style

set_plot_style()


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_features(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """log1p-transform then standardise the RFM features.

    Why: K-Means uses Euclidean distance. Raw Monetary values are in the
    thousands while Frequency is usually below 10, so without scaling Monetary
    would dominate the distance. The log transform first reduces the strong
    right skew so a handful of extreme customers do not pull the centroids.
    """
    log_features = np.log1p(rfm[RFM_FEATURES])
    scaler = StandardScaler()
    X = scaler.fit_transform(log_features)
    return X, scaler


# ---------------------------------------------------------------------------
# Choosing k
# ---------------------------------------------------------------------------
def evaluate_k_range(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    """Fit K-Means for each k and record inertia (elbow), silhouette and Davies-Bouldin."""
    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })
    return pd.DataFrame(rows).round(4)


def find_elbow(k_values, inertia) -> int:
    """Elbow = the point farthest from the straight line joining the first and last points."""
    k = np.asarray(k_values, dtype=float)
    y = np.asarray(inertia, dtype=float)
    # Normalise both axes to [0, 1] so the distance is not dominated by inertia's scale.
    x_n = (k - k.min()) / (k.max() - k.min())
    y_n = (y - y.min()) / (y.max() - y.min())
    x1, y1, x2, y2 = x_n[0], y_n[0], x_n[-1], y_n[-1]
    # Perpendicular distance of every point from the line (x1, y1) -> (x2, y2)
    distances = np.abs((y2 - y1) * x_n - (x2 - x1) * y_n + x2 * y1 - y2 * x1)
    return int(k[np.argmax(distances)])


def select_k(eval_df: pd.DataFrame, min_k: int = 3) -> int:
    """Choose the k with the highest silhouette score among k >= min_k.

    k = 2 often has the highest silhouette on RFM data because it simply
    splits 'active' from 'inactive' customers. That is too coarse to be
    useful for segmentation, so the search starts at min_k = 3. The elbow
    point is reported alongside as a second opinion.
    """
    candidates = eval_df[eval_df["k"] >= min_k]
    if candidates.empty:
        raise ValueError(f"No k >= {min_k} was evaluated.")
    return int(candidates.loc[candidates["silhouette"].idxmax(), "k"])


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_kmeans(X: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    model.fit(X)
    return model


def stability_check(X: np.ndarray, k: int, seeds=(0, 1, 2, 3, 4)) -> float:
    """Average Adjusted Rand Index between runs with different random seeds.

    Values close to 1 mean the clustering is reproducible and not an
    accident of the random initialisation.
    """
    runs = [KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(X) for s in seeds]
    scores = [adjusted_rand_score(runs[i], runs[j])
              for i in range(len(runs)) for j in range(i + 1, len(runs))]
    return float(np.mean(scores))


def apply_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    """Project scaled features to 2D for visualisation only."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    return coords, pca


def save_models(model: KMeans, scaler: StandardScaler, pca: PCA,
                models_dir: Path = MODELS_DIR) -> None:
    ensure_dirs(models_dir)
    joblib.dump(model, models_dir / "kmeans_model.joblib")
    joblib.dump(scaler, models_dir / "rfm_scaler.joblib")
    joblib.dump(pca, models_dir / "pca_model.joblib")


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
SEGMENT_ACTIONS = {
    "High-Value Customers": "Retain with loyalty rewards, early access and personal service.",
    "Loyal / Frequent Customers": "Encourage larger baskets with bundles and cross-selling.",
    "Potential Customers": "Recently active with average or light buying: nurture with recommendations to grow frequency and basket size.",
    "At-Risk Customers": "Used to buy well but have gone quiet: send win-back campaigns.",
    "Low-Engagement Customers": "Low value and inactive: use low-cost, automated re-engagement only.",
}


# A cluster mean within +/-0.25 standard deviations is treated as "about average".
AVERAGE_BAND = 0.25
HIGH = 0.5


def _name_segment(r: float, f: float, m: float) -> str:
    """Rule-based name from a cluster's mean standardised scores.

    r, f, m are cluster means of the standardised log features, with the
    recency sign flipped so that a higher value always means 'better'.
    0 is the average customer.
    """
    if r > 0 and f > HIGH and m > HIGH:
        return "High-Value Customers"
    if r > 0 and (f > AVERAGE_BAND or m > AVERAGE_BAND):
        return "Loyal / Frequent Customers"
    if r > 0:
        return "Potential Customers"
    if f > AVERAGE_BAND or m > AVERAGE_BAND:
        return "At-Risk Customers"
    return "Low-Engagement Customers"


def interpret_clusters(rfm: pd.DataFrame, X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Summarise each cluster and name it from its actual RFM statistics.

    The names are analytical interpretations, not ground-truth categories.
    """
    scores = pd.DataFrame(X, columns=RFM_FEATURES)
    scores["Cluster"] = labels
    z_means = scores.groupby("Cluster")[RFM_FEATURES].mean()

    data = rfm.assign(Cluster=labels)
    profile = data.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        Recency_mean=("Recency", "mean"),
        Recency_median=("Recency", "median"),
        Frequency_mean=("Frequency", "mean"),
        Frequency_median=("Frequency", "median"),
        Monetary_mean=("Monetary", "mean"),
        Monetary_median=("Monetary", "median"),
        Revenue=("Monetary", "sum"),
    )
    # Round once here, so tables, text and plots all show the same value.
    profile["CustomerSharePct"] = (profile["Customers"] / profile["Customers"].sum() * 100).round(1)
    profile["RevenueSharePct"] = (profile["Revenue"] / profile["Revenue"].sum() * 100).round(1)
    profile["R_z"] = -z_means["Recency"]  # flipped: higher = more recent
    profile["F_z"] = z_means["Frequency"]
    profile["M_z"] = z_means["Monetary"]

    profile["Segment"] = [
        _name_segment(row.R_z, row.F_z, row.M_z) for row in profile.itertuples()
    ]
    # If two clusters receive the same name, keep both but make them distinguishable.
    duplicated = profile["Segment"].duplicated(keep=False)
    profile.loc[duplicated, "Segment"] = (
        profile.loc[duplicated, "Segment"] + " (Cluster " + profile.index[duplicated].astype(str) + ")"
    )

    profile["Description"] = [_describe(row) for row in profile.itertuples()]
    base_names = profile["Segment"].str.replace(r" \(Cluster \d+\)", "", regex=True)
    profile["SuggestedAction"] = base_names.map(SEGMENT_ACTIONS)
    return profile.round(2).reset_index()


def _describe(row) -> str:
    def compare(value, good, bad):
        if abs(value) < AVERAGE_BAND:
            return "about average"
        return good if value > 0 else bad

    return (
        f"{row.Customers:,} customers ({row.CustomerSharePct:.1f}% of customers, {row.RevenueSharePct:.1f}% of revenue). "
        f"Median customer last bought {row.Recency_median:.0f} days ago, placed "
        f"{row.Frequency_median:.0f} order(s) and spent {CURRENCY}{row.Monetary_median:,.0f}. "
        f"Compared with the average customer: recency is "
        f"{compare(row.R_z, 'more recent', 'older')}, frequency is "
        f"{compare(row.F_z, 'higher', 'lower')} and spending is "
        f"{compare(row.M_z, 'higher', 'lower')}."
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_k_evaluation(eval_df: pd.DataFrame, selected_k: int, elbow_k: int,
                      fig_dir: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(eval_df["k"], eval_df["inertia"], marker="o")
    axes[0].axvline(elbow_k, color="grey", linestyle="--", label=f"Elbow (k={elbow_k})")
    axes[0].set(title="Elbow Method", xlabel="Number of clusters (k)",
                ylabel="Inertia (within-cluster SSE)")
    axes[0].legend()

    axes[1].plot(eval_df["k"], eval_df["silhouette"], marker="o", color="tab:green")
    axes[1].axvline(selected_k, color="red", linestyle="--", label=f"Selected k={selected_k}")
    axes[1].set(title="Silhouette Score (higher is better)",
                xlabel="Number of clusters (k)", ylabel="Silhouette score")
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, "10_k_selection.png", fig_dir)
    return fig


def plot_pca_clusters(coords: np.ndarray, segments: pd.Series, explained: np.ndarray,
                      fig_dir: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=segments.values,
                    hue_order=sorted(segments.unique()), s=14, alpha=0.6, ax=ax, palette="deep")
    ax.set(
        title=f"Customer Segments in PCA Space ({explained.sum() * 100:.1f}% variance shown)",
        xlabel=f"PC1 ({explained[0] * 100:.1f}% variance)",
        ylabel=f"PC2 ({explained[1] * 100:.1f}% variance)",
    )
    ax.legend(title="Segment", fontsize=8, markerscale=2)
    save_figure(fig, "11_pca_clusters.png", fig_dir)
    return fig


def plot_segment_comparison(profile: pd.DataFrame, fig_dir: Path | None = None) -> plt.Figure:
    """Median Recency, Frequency and Monetary per segment side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    columns = [("Recency_median", "Median Recency (days, lower = better)"),
               ("Frequency_median", "Median Frequency (orders)"),
               ("Monetary_median", f"Median Monetary ({CURRENCY})")]
    order = profile.sort_values("Monetary_median", ascending=False)
    hue_order = sorted(profile["Segment"])  # same colours as the PCA plot
    for ax, (col, title) in zip(axes, columns):
        sns.barplot(data=order, y="Segment", x=col, hue="Segment", hue_order=hue_order,
                    legend=False, ax=ax, palette="deep")
        ax.set(title=title, xlabel="", ylabel="")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Segment-level RFM Comparison", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "12_segment_rfm_comparison.png", fig_dir)
    return fig
