"""Gene importances for the four condition classifiers, computed from each model's native measure.

    Logistic regression : mean |coefficient| across the three classes (features are standardised)
    Random forest       : impurity decrease (Gini), summed over all trees, sums to 1
    XGBoost             : gain (average loss reduction per split using the gene), normalised to sum to 1
    Neural network      : permutation importance (a network has no native importance), read from
                          results/nn/importances.csv, written by train_nn.py

Native importances are read from the saved model.pkl files, nothing is retrained, except the random
forest when its model.pkl is missing: pass --refit-rf to refit it with train_random_forest.py's
settings (deterministic; the refit must reproduce the test accuracy in results/random_forest/metrics.json).

Outputs (in --out-dir):
    importance_summary.csv      every gene: raw importance and rank per model, per-class logreg
                                coefficients, mean expression per condition
    importance_top_genes.png    top genes of each model, side by side
    importance_heatmap.png      the same genes across all models (cell = relative importance, text = rank)

Usage:
    python feature_importance.py                 # models that have a model.pkl
    python feature_importance.py --refit-rf      # also refit + save the random forest first
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from data_prep import load_features, make_split

COLORS = {
    "Logistic regression": "#0072B2",
    "Random forest": "#009E73",
    "XGBoost": "#D55E00",
    "Neural network": "#CC79A7",
}
XLABELS = {
    "Logistic regression": "Mean |coefficient|",
    "Random forest": "Impurity decrease",
    "XGBoost": "Gain",
    "Neural network": "Permutation importance",
}


def logreg_importance(results_dir, genes):
    with open(os.path.join(results_dir, "logreg", "model.pkl"), "rb") as f:
        d = pickle.load(f)
    assert list(d["gene_names"]) == list(genes), "logreg gene order differs from the data"
    coef = pd.DataFrame(d["model"].coef_.T, index=genes, columns=[f"coef_{c}" for c in d["model"].classes_])
    return np.abs(d["model"].coef_).mean(axis=0), coef


def xgboost_importance(results_dir, genes):
    with open(os.path.join(results_dir, "xgboost", "model.pkl"), "rb") as f:
        d = pickle.load(f)
    assert list(d["gene_names"]) == list(genes), "xgboost gene order differs from the data"
    return np.asarray(d["model"].feature_importances_, dtype=float)  # normalised gain


def refit_rf(results_dir, X, y, seed=0, n_estimators=500):
    """Refit the random forest with train_random_forest.py's settings; must reproduce its test accuracy."""
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test, _ = make_split(X, y, seed=seed)
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed,
                                 class_weight="balanced_subsample")
    print(f"Refitting random forest on {X_train.shape} (single-core machines: expect 20-30 min)...", flush=True)
    t0 = time.time()
    clf.fit(X_train, y_train)
    acc = float(np.mean(clf.predict(X_test) == y_test))
    with open(os.path.join(results_dir, "random_forest", "metrics.json")) as f:
        ref = json.load(f)["accuracy"]
    print(f"  fitted in {time.time() - t0:.0f}s, test accuracy {acc:.6f} (metrics.json: {ref:.6f})", flush=True)
    if not np.isclose(acc, ref, atol=1e-4):
        raise RuntimeError("Refit random forest does not reproduce the saved test accuracy; not saving it.")
    with open(os.path.join(results_dir, "random_forest", "model.pkl"), "wb") as f:
        pickle.dump({"model": clf}, f)


def rf_importance(results_dir):
    path = os.path.join(results_dir, "random_forest", "model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f)["model"].feature_importances_, dtype=float)


def plot_top_genes(imp, out_path, top_n=15):
    models = list(imp.columns)
    tops = {m: imp[m].nlargest(top_n) for m in models}
    counts = pd.Series([g for t in tops.values() for g in t.index]).value_counts()
    shared = set(counts[counts >= 2].index)

    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 5.4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        t = tops[m].iloc[::-1]
        colors = [COLORS[m] if g in shared else matplotlib.colors.to_rgba(COLORS[m], 0.3) for g in t.index]
        ax.barh(range(len(t)), t.values, color=colors, height=0.7)
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels(t.index, fontsize=10)
        ax.set_xlabel(XLABELS[m], fontsize=10)
        ax.set_title(m, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.legend(
        [Patch(color="#555555"), Patch(color="#555555", alpha=0.3)],
        ["Also in another model's top genes", "Only in this model's top genes"],
        loc="lower center", ncol=2, frameon=False, fontsize=10,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(imp, ranks, out_path, top_n=15):
    models = list(imp.columns)
    union = set()
    for m in models:
        union |= set(imp[m].nlargest(top_n).index)
    genes = ranks.loc[list(union), models].mean(axis=1).sort_values().index
    rel = (imp / imp.max()).loc[genes]

    fig, ax = plt.subplots(figsize=(1.5 * len(models) + 2, 0.3 * len(genes) + 1.6))
    im = ax.imshow(rel.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=10)
    for i, g in enumerate(genes):
        for j, m in enumerate(models):
            v = rel.loc[g, m]
            ax.text(j, i, f"{int(ranks.loc[g, m])}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 else "#333333")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Relative importance (top gene = 1)", fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title("Numbers show each gene's rank in that model", fontsize=10, pad=34)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(results_dir, out_dir, top_n, do_refit_rf):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading features...", flush=True)
    X, y, genes = load_features()

    if do_refit_rf:
        refit_rf(results_dir, X, y)

    imp = pd.DataFrame(index=pd.Index(genes, name="gene"))
    imp["Logistic regression"], coef = logreg_importance(results_dir, genes)
    rf = rf_importance(results_dir)
    if rf is None:
        print("[skip] random forest: results/random_forest/model.pkl not found (use --refit-rf)")
    else:
        imp["Random forest"] = rf
    imp["XGBoost"] = xgboost_importance(results_dir, genes)
    nn = pd.read_csv(os.path.join(results_dir, "nn", "importances.csv"), index_col="gene")
    imp["Neural network"] = nn["perm_importance_mean"].reindex(genes).to_numpy()
    assert imp.notna().all().all(), "missing importance values"

    ranks = imp.rank(ascending=False, method="min")

    # descriptive: mean log-normalised expression per condition over all cells
    expr = pd.DataFrame({f"mean_expr_{c}": X[y == c].mean(axis=0) for c in np.unique(y)}, index=imp.index)

    summary = pd.concat(
        [imp.add_prefix("importance_"), ranks.add_prefix("rank_"), coef, expr], axis=1
    )
    summary["mean_rank"] = ranks.mean(axis=1)
    summary = summary.sort_values("mean_rank")
    summary.to_csv(os.path.join(out_dir, "importance_summary.csv"))

    plot_top_genes(imp, os.path.join(out_dir, "importance_top_genes.png"), top_n)
    plot_heatmap(imp, ranks, os.path.join(out_dir, "importance_heatmap.png"), top_n)

    print(f"Saved to {out_dir}")
    for m in imp.columns:
        print(f"\nTop {top_n} by {XLABELS[m].lower()} ({m}):")
        print(", ".join(imp[m].nlargest(top_n).index))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/importance")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--refit-rf", action="store_true",
                        help="refit + save the random forest (needed if results/random_forest/model.pkl is missing)")
    args = parser.parse_args()
    main(args.results_dir, args.out_dir, args.top_n, args.refit_rf)
