import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import umap
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

SIGNATURES = "results/signatures.csv"
FIGDIR = "results/figures"
N_PCS = 30
K_RANGE = range(4, 26)
DEMO_K = 4  # illustrative k used only for the magnitude-vs-normalized demo
RANDOM_STATE = 0

METHOD_COLOR = {"hierarchical": "#4C72B0", "kmeans": "#DD8452", "leiden": "#55A868"}


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def row_standardize(X: np.ndarray) -> np.ndarray:
    """Per-row (per-perturbation) z-score across genes. See Section 2 in the
    notebook / the module docstring above for why: this is what makes
    Euclidean distance on the result an exact monotonic function of
    (1 - Pearson correlation) between the corresponding raw rows."""
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)


def column_standardize(X: np.ndarray) -> np.ndarray:
    """Per-column (per-gene) z-score across perturbations -- the 'do nothing
    about magnitude' baseline used only for the Section 2 demonstration."""
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def pca_embed(X: np.ndarray, n_pcs: int) -> np.ndarray:
    n_pcs = min(n_pcs, min(X.shape) - 1)
    return PCA(n_components=n_pcs, random_state=RANDOM_STATE).fit_transform(X)


def best_by_silhouette(X, cluster_fn, k_range):
    scores = {}
    labels_by_k = {}
    for k in k_range:
        labels = cluster_fn(k)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(X, labels)
        labels_by_k[k] = labels
    best_k = max(scores, key=scores.get)
    return best_k, labels_by_k[best_k], scores

#---------


def fig_correlation_map(sig: pd.DataFrame):
    corr = sig.T.corr() 
    g = sns.clustermap(
        corr,
        method="average",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(14, 14),
        xticklabels=False,
        yticklabels=False,
        dendrogram_ratio=0.1,
        cbar_pos=(0.02, 0.83, 0.03, 0.15),
    )
    g.ax_heatmap.set_xlabel("Perturbation")
    g.ax_heatmap.set_ylabel("Perturbation")
    g.fig.suptitle(
        "Perturbation signature correlation -- 239 Control-condition knockouts (initial look)",
        y=1.02, fontsize=13,
    )
    g.savefig(f"{FIGDIR}/signature_correlation_map.png", dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    return corr


def fig_magnitude_vs_normalized(sig: pd.DataFrame):
    norms = pd.Series(np.linalg.norm(sig.values, axis=1), index=sig.index).sort_values(ascending=False)

    X_mag = pca_embed(column_standardize(sig.values), N_PCS)
    X_corr = pca_embed(row_standardize(sig.values), N_PCS)

    Z_mag = linkage(X_mag, method="ward")
    Z_corr = linkage(X_corr, method="ward")
    labels_mag = fcluster(Z_mag, t=DEMO_K, criterion="maxclust")
    labels_corr = fcluster(Z_corr, t=DEMO_K, criterion="maxclust")

    sizes_mag = pd.Series(labels_mag).value_counts().sort_values(ascending=False)
    sizes_corr = pd.Series(labels_corr).value_counts().sort_values(ascending=False)

    outlier_cluster = sizes_mag.index[1:]  # everything but the single largest cluster
    outlier_genes = sig.index[np.isin(labels_mag, outlier_cluster)].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    top_n = 20
    top_norms = norms.head(top_n)
    axes[0].barh(range(top_n), top_norms.values[::-1], color="#4C72B0")
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_norms.index[::-1], fontsize=8)
    axes[0].set_xlabel("‖signature‖₂ (2000 HVGs)")
    axes[0].set_title(f"Top {top_n} perturbations by raw\nsignature magnitude")

    axes[1].bar(range(len(sizes_mag)), sizes_mag.values, color="#C44E52")
    axes[1].set_xticks(range(len(sizes_mag)))
    axes[1].set_xticklabels([f"cluster {i+1}" for i in range(len(sizes_mag))])
    axes[1].set_ylabel("# perturbations")
    axes[1].set_title(f"Ward, k={DEMO_K}, on RAW-MAGNITUDE\n(column-standardized) signatures")
    for i, v in enumerate(sizes_mag.values):
        axes[1].text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    axes[2].bar(range(len(sizes_corr)), sizes_corr.values, color="#55A868")
    axes[2].set_xticks(range(len(sizes_corr)))
    axes[2].set_xticklabels([f"cluster {i+1}" for i in range(len(sizes_corr))])
    axes[2].set_ylabel("# perturbations")
    axes[2].set_title(f"Ward, k={DEMO_K}, on ROW-NORMALIZED\n(correlation-equivalent) signatures")
    for i, v in enumerate(sizes_corr.values):
        axes[2].text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Why row-normalize before clustering? Raw magnitude is dominated by a few huge-effect knockouts",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/magnitude_vs_normalized_clustering.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return norms, sizes_mag, sizes_corr, outlier_genes

def run_hierarchical(X, k_range):
    Z = linkage(X, method="ward")
    best_k, labels, scores = best_by_silhouette(
        X, lambda k: fcluster(Z, t=k, criterion="maxclust"), k_range
    )
    return labels, best_k, scores


def run_kmeans(X, k_range):
    best_k, labels, scores = best_by_silhouette(
        X, lambda k: KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X), k_range
    )
    return labels, best_k, scores


def run_leiden(sig_index, X_pca, resolutions=(0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0)):
    ad = sc.AnnData(X_pca.astype(np.float32))
    ad.obs_names = sig_index
    ad.obsm["X_pca"] = X_pca
    sc.pp.neighbors(ad, use_rep="X_pca", random_state=RANDOM_STATE)

    labels_by_res = {}
    sil_by_res = {}
    for res in resolutions:
        key = f"leiden_{res}"
        sc.tl.leiden(ad, resolution=res, key_added=key, random_state=RANDOM_STATE, flavor="igraph", n_iterations=2)
        labels = ad.obs[key].astype(int).values
        labels_by_res[res] = labels
        sil_by_res[res] = silhouette_score(X_pca, labels) if len(set(labels)) >= 2 else np.nan

    best_res = max(sil_by_res, key=sil_by_res.get)
    return labels_by_res[best_res], best_res, sil_by_res, labels_by_res


def fig_silhouette_selection(hier_scores, km_scores, leiden_sil_by_res, hier_k, km_k, leiden_res):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, scores, chosen, title, color in [
        (axes[0], hier_scores, hier_k, "Hierarchical (Ward)", METHOD_COLOR["hierarchical"]),
        (axes[1], km_scores, km_k, "K-means", METHOD_COLOR["kmeans"]),
    ]:
        ks = sorted(scores)
        vals = [scores[k] for k in ks]
        ax.plot(ks, vals, marker="o", color=color, markersize=4)
        ax.axvline(chosen, color="black", linestyle="--", linewidth=1)
        ax.scatter([chosen], [scores[chosen]], color="black", zorder=5, s=50)
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette score")
        ax.set_title(f"{title}\nchosen k={chosen}")

    ax = axes[2]
    resolutions = sorted(leiden_sil_by_res)
    vals = [leiden_sil_by_res[r] for r in resolutions]
    ax.plot(resolutions, vals, marker="o", color=METHOD_COLOR["leiden"], markersize=4)
    ax.axvline(leiden_res, color="black", linestyle="--", linewidth=1)
    ax.scatter([leiden_res], [leiden_sil_by_res[leiden_res]], color="black", zorder=5, s=50)
    ax.set_xlabel("resolution")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"Leiden\nchosen resolution={leiden_res}")

    fig.suptitle("Silhouette-based selection of cluster count / resolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/silhouette_selection.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def fig_umap(sig_index, X_pca, all_labels, method_param):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=RANDOM_STATE)
    emb = reducer.fit_transform(X_pca)
    emb_df = pd.DataFrame(emb, index=sig_index, columns=["UMAP1", "UMAP2"])

    methods = ["hierarchical", "kmeans", "leiden"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, method in zip(axes, methods):
        lab = pd.Series(all_labels[method], index=sig_index)
        cmap = sns.color_palette("husl", lab.nunique())
        lut = dict(zip(sorted(lab.unique()), cmap))
        colors = lab.map(lut)
        ax.scatter(emb_df["UMAP1"], emb_df["UMAP2"], c=colors, s=20, linewidths=0)
        ax.set_title(f"{method} ({method_param[method]})", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Perturbation-signature UMAP (shared layout, row-normalized signatures) -- clustering methods compared",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/umap_clustering_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    sig = pd.read_csv(SIGNATURES, index_col=0)
    genes = sig.index
    print(f"Loaded {sig.shape[0]} perturbations x {sig.shape[1]} HVGs")

    # 1. Correlation map
    fig_correlation_map(sig)
    print("Saved signature_correlation_map.png")

    # 2. Magnitude vs. normalized demo
    norms, sizes_mag, sizes_corr, outlier_genes = fig_magnitude_vs_normalized(sig)
    print("Saved magnitude_vs_normalized_clustering.png")
    print(f"  Raw-magnitude Ward (k={DEMO_K}) cluster sizes: {sizes_mag.tolist()}")
    print(f"  Row-normalized Ward (k={DEMO_K}) cluster sizes: {sizes_corr.tolist()}")

    # 3. Clustering on row-normalized signatures
    X_corr = row_standardize(sig.values)
    X_pca = pca_embed(X_corr, N_PCS)

    hier_labels, hier_k, hier_scores = run_hierarchical(X_pca, K_RANGE)
    km_labels, km_k, km_scores = run_kmeans(X_pca, K_RANGE)
    leiden_labels, leiden_res, leiden_sil_by_res, leiden_labels_by_res = run_leiden(genes, X_pca)
    leiden_k = len(set(leiden_labels))

    all_labels = {"hierarchical": hier_labels, "kmeans": km_labels, "leiden": leiden_labels}
    # Leiden is parameterized by resolution, not k -- label it that way in plots
    method_param = {"hierarchical": f"k={hier_k}", "kmeans": f"k={km_k}", "leiden": f"resolution={leiden_res}"}

    fig_silhouette_selection(hier_scores, km_scores, leiden_sil_by_res, hier_k, km_k, leiden_res)
    print("Saved silhouette_selection.png")
    print(f"  Chosen: hierarchical k={hier_k}, kmeans k={km_k}, leiden resolution={leiden_res} (k={leiden_k})")

    # 4. UMAP
    fig_umap(genes, X_pca, all_labels, method_param)
    print("Saved umap_clustering_comparison.png")

    # Save cluster labels + selection scores
    labels_df = pd.DataFrame(all_labels, index=genes)
    labels_df.to_csv("results/cluster_labels.csv")

    with open("results/k_selection_scores.json", "w") as f:
        json.dump(
            {
                "hierarchical_silhouette_by_k": {str(k): v for k, v in hier_scores.items()},
                "kmeans_silhouette_by_k": {str(k): v for k, v in km_scores.items()},
                "leiden_silhouette_by_resolution": {str(r): v for r, v in leiden_sil_by_res.items()},
                "leiden_k_by_resolution": {str(r): int(len(set(l))) for r, l in leiden_labels_by_res.items()},
                "chosen_k": {"hierarchical": hier_k, "kmeans": km_k, "leiden": leiden_k},
                "chosen_leiden_resolution": leiden_res,
                "magnitude_demo_k": DEMO_K,
                "magnitude_demo_cluster_sizes_raw": sizes_mag.tolist(),
                "magnitude_demo_cluster_sizes_normalized": sizes_corr.tolist(),
                "magnitude_demo_outlier_genes": outlier_genes,
            },
            f,
            indent=2,
        )

    print("\nSaved: cluster_labels.csv, k_selection_scores.json")
    print("Figures in:", FIGDIR)


if __name__ == "__main__":
    main()
