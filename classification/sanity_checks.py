"""Negative controls for the condition classifier: is ~98% accuracy too good to be true?

Two controls, both using the same features and the same cached train/test split as the four models:

1. Label shuffle: fit logistic regression on randomly permuted *training* labels, score on the real
   test labels. If the pipeline leaks information, this would still score high. Expect ~chance.
2. Gene-subset scaling: fit logistic regression on k genes chosen (a) at random from the 2000 HVGs
   and (b) as the top-k by a ranking derived from the *training set only* (mean |coefficient| of a
   logistic regression fit on the training data). Compare accuracy vs. k.

Note on scope: "random" genes here are drawn from the 2000 highly variable genes, not from the whole
transcriptome, so this says nothing about genes outside the HVG panel.

Usage:
    python sanity_checks.py --results-dir results/sanity_checks
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from data_prep import load_features, make_split

BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#999999"


def fit_eval(X_tr, y_tr, X_te, y_te, seed=0, max_iter=1000):
    clf = LogisticRegression(solver="lbfgs", max_iter=max_iter, random_state=seed)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return clf, {
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
    }


def make_plot(res, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})

    # --- left: label shuffle ---
    shuf = [r["accuracy"] for r in res["label_shuffle"]]
    heights = [res["majority_baseline_accuracy"], float(np.mean(shuf)), res["full_model"]["accuracy"]]
    bars = ax1.bar(range(3), heights, color=[GRAY, ORANGE, BLUE], width=0.6)
    for b, h in zip(bars, heights):
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", fontsize=10)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(["Majority\nclass", "Shuffled\nlabels", "Real\nlabels"], fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Test accuracy")
    ax1.set_title("Shuffled labels", fontsize=12)
    ax1.spines[["top", "right"]].set_visible(False)

    # --- right: random vs. top genes ---
    ks = res["gene_counts"]
    cur = [res["curated"][str(k)]["accuracy"] for k in ks]
    rnd = {k: [r["accuracy"] for r in res["random"][str(k)]] for k in ks}
    ax2.fill_between(ks, [min(rnd[k]) for k in ks], [max(rnd[k]) for k in ks], color=ORANGE, alpha=0.2, lw=0,
                     label="Random genes (min–max)")
    for k in ks:
        ax2.scatter(np.full(len(rnd[k]), k), rnd[k], s=10, c=ORANGE, alpha=0.4, linewidths=0, zorder=2)
    ax2.plot(ks, [np.mean(rnd[k]) for k in ks], "-o", color=ORANGE, lw=2, ms=5, label="Random genes (mean)")
    ax2.plot(ks, cur, "-o", color=BLUE, lw=2, ms=5, label="Top genes")
    ax2.axhline(res["full_model"]["accuracy"], color=BLUE, ls="--", lw=1, label="All 2000 genes")
    ax2.axhline(res["majority_baseline_accuracy"], color=GRAY, ls="--", lw=1, label="Majority class")
    ax2.set_xscale("log")
    ax2.set_xticks(ks)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.minorticks_off()
    ax2.set_ylim(0.3, 1.02)
    ax2.set_xlabel("Number of genes")
    ax2.set_ylabel("Test accuracy")
    ax2.set_title("Random vs. top genes", fontsize=12)
    ax2.spines[["top", "right"]].set_visible(False)
    # legend sits below the axes so it can never cover the data
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=9,
               columnspacing=1.2, handlelength=1.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(results_dir, gene_counts, n_random, n_shuffle, seed):
    os.makedirs(results_dir, exist_ok=True)
    t0 = time.time()

    print("Loading features...", flush=True)
    X, y, gene_names = load_features()
    X_train, X_test, y_train, y_test, _ = make_split(X, y, seed=seed)
    print(f"train={X_train.shape} test={X_test.shape}", flush=True)

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    vals, counts = np.unique(y_train, return_counts=True)
    majority = vals[np.argmax(counts)]
    res = {
        "n_train": int(len(y_train)), "n_test": int(len(y_test)), "n_genes": int(X.shape[1]),
        "majority_class": str(majority),
        "majority_baseline_accuracy": float(np.mean(y_test == majority)),
        "gene_counts": list(gene_counts),
    }

    print("Fitting reference model on all genes...", flush=True)
    full_clf, res["full_model"] = fit_eval(Xtr, y_train, Xte, y_test, seed=seed)
    print(f"  all genes: {res['full_model']}", flush=True)
    # training-set-only ranking of genes: mean |coef| across the three classes
    order = np.argsort(-np.abs(full_clf.coef_).mean(axis=0))
    res["top_genes_train_ranking"] = [str(g) for g in gene_names[order[:max(gene_counts)]]]

    print("Control 1: label shuffle", flush=True)
    res["label_shuffle"] = []
    for s in range(n_shuffle):
        y_shuf = np.random.RandomState(s).permutation(y_train)
        _, m = fit_eval(Xtr, y_shuf, Xte, y_test, seed=seed)
        m["shuffle_seed"] = s
        res["label_shuffle"].append(m)
        print(f"  seed {s}: {m}", flush=True)

    print("Control 2: gene subsets", flush=True)
    res["curated"], res["random"] = {}, {}
    for k in gene_counts:
        _, res["curated"][str(k)] = fit_eval(Xtr[:, order[:k]], y_train, Xte[:, order[:k]], y_test, seed=seed)
        res["random"][str(k)] = []
        for t in range(n_random):
            idx = np.random.RandomState(10_000 * k + t).choice(X.shape[1], size=k, replace=False)
            _, m = fit_eval(Xtr[:, idx], y_train, Xte[:, idx], y_test, seed=seed)
            m["trial"] = t
            res["random"][str(k)].append(m)
        accs = [r["accuracy"] for r in res["random"][str(k)]]
        print(f"  k={k:4d}  curated={res['curated'][str(k)]['accuracy']:.4f}  "
              f"random mean={np.mean(accs):.4f} (min {min(accs):.4f}, max {max(accs):.4f})", flush=True)

    with open(os.path.join(results_dir, "sanity_checks.json"), "w") as f:
        json.dump(res, f, indent=2)
    make_plot(res, os.path.join(results_dir, "sanity_checks.png"))
    print(f"Done in {time.time() - t0:.0f}s. Saved to {results_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/sanity_checks")
    parser.add_argument("--gene-counts", type=int, nargs="+", default=[10, 25, 50, 100, 200, 500])
    parser.add_argument("--n-random", type=int, default=10, help="random gene subsets per size")
    parser.add_argument("--n-shuffle", type=int, default=5, help="label-shuffle repetitions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-only", action="store_true", help="redraw the plot from the saved JSON, no refitting")
    args = parser.parse_args()
    if args.plot_only:
        with open(os.path.join(args.results_dir, "sanity_checks.json")) as f:
            make_plot(json.load(f), os.path.join(args.results_dir, "sanity_checks.png"))
        raise SystemExit(0)
    main(args.results_dir, args.gene_counts, args.n_random, args.n_shuffle, args.seed)
