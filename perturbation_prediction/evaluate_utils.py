import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

SEED = 0
COLORS = {"baseline": "#444444", "knn_raw": "#0072B2", "knn_scalars": "#56B4E9",
          "mlp_raw": "#D55E00", "mlp_scalars": "#E69F00"}


def pearson(pred, obs):
    if np.ptp(pred) == 0 or np.ptp(obs) == 0:
        return 0.0
    return float(pearsonr(pred, obs)[0])


def rmse(pred, obs):
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def bootstrap_ci(cache_counts, cache_pert, prior, perturbation, pred, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    ctrl = cache_counts[cache_pert == "control"].mean(0)
    idx = np.where(cache_pert == perturbation)[0]
    pear = np.empty(n_boot)
    rms = np.empty(n_boot)
    for b in range(n_boot):
        bs = rng.choice(idx, size=len(idx), replace=True)
        obs = np.log2((cache_counts[bs].mean(0) + prior) / (ctrl + prior))
        pear[b] = pearson(pred, obs)
        rms[b] = rmse(pred, obs)
    ci = lambda v: (float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return ci(pear), ci(rms)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def plot_summary_bar(names, means, ci_los, ci_his, out_path, title, ylabel, ylim=None):
    """Mean metric per model, with a bootstrap CI on the mean (whiskers)."""
    means, ci_los, ci_his = map(np.asarray, (means, ci_los, ci_his))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    yerr = np.vstack([means - ci_los, ci_his - means])
    ax.bar(names, means, color=[COLORS.get(n, "#ccc") for n in names], width=0.6)
    ax.errorbar(names, means, yerr=yerr, fmt="none", ecolor="#222", capsize=4, lw=1.2)
    top = ylim[1] if (ylim and ylim[1] is not None) else float(ci_his.max())
    pad = 0.02 * top
    for i, (m, hi) in enumerate(zip(means, ci_his)):
        ax.text(i, hi + pad, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10, wrap=True)
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_perturbation_ci(per_model, out_path, title, metric="pearson", xlabel=None):
    models = list(per_model)
    perts = list(next(iter(per_model.values())).index)
    n = len(perts)
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.32 * n)))
    offsets = np.linspace(-0.3, 0.3, len(models))
    for mo, dx in zip(models, offsets):
        d = per_model[mo]
        y = np.arange(n) + dx
        col = COLORS.get(mo, "#888")
        point = d[f"{metric}_boot_mean"] if f"{metric}_boot_mean" in d else d[metric]
        ax.hlines(y, d[f"{metric}_ci_lo"].to_numpy(), d[f"{metric}_ci_hi"].to_numpy(),
                  color=col, lw=1.4, alpha=0.8)
        ax.plot(point.to_numpy(), y, "o", ms=4, color=col, label=mo)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(perts, fontsize=8)
    ax.set_xlabel(xlabel or f"{metric} (predicted vs observed log2FC)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="lower left")
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_panels(model_name, pred_df, obs_df, out_path, ncols=5):
    perts = list(pred_df.index)
    nrows = int(np.ceil(len(perts) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    for ax, p in zip(axes.ravel(), perts):
        o, pr = obs_df.loc[p].to_numpy(), pred_df.loc[p].to_numpy()
        ax.scatter(o, pr, s=6, alpha=0.3, color=COLORS.get(model_name, "#0072B2"))
        lim = np.abs(o).max() * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.6)
        ax.set_title(f"{p}  r={pearson(pr, o):.2f}", fontsize=9)
        ax.set_xlabel("observed", fontsize=8)
        ax.set_ylabel("predicted", fontsize=8)
    for ax in axes.ravel()[len(perts):]:
        ax.axis("off")
    fig.suptitle(f"{model_name}: predicted vs observed log2FC (held-out test perturbations)", y=1.002)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
