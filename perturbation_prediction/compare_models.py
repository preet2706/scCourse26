import json
import os

import numpy as np
import pandas as pd

import evaluate_utils as ev
import data_prep as dp

RESULTS = dp.RESULTS
CMP = os.path.join(RESULTS, "comparison")
ORDER = ["baseline", "knn_raw", "knn_scalars", "mlp_raw", "mlp_scalars"]


def _load(model):
    d = os.path.join(RESULTS, model)
    f = os.path.join(d, "metrics_test.csv")
    if not os.path.exists(f):
        return None
    return {"metrics_test": pd.read_csv(f, index_col=0),
            "pred_test": pd.read_csv(os.path.join(d, "predictions_test.csv"), index_col=0),
            "summary": json.load(open(os.path.join(d, "metrics.json")))}


def _mean_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    os.makedirs(CMP, exist_ok=True)
    models = {m: _load(m) for m in ORDER}
    models = {m: d for m, d in models.items() if d is not None}
    targets = dp.load_targets()
    split = dp.load_split()

    rows = []
    agg = {}
    for m, d in models.items():
        mt = d["metrics_test"]
        row = {"model": m, "features": d["summary"]["features"]}
        for metric in ("pearson", "rmse"):
            mean, lo, hi = _mean_ci(mt[metric])
            row.update({f"{metric}_mean": mean, f"{metric}_ci_lo": lo, f"{metric}_ci_hi": hi,
                        f"{metric}_median": mt[metric].median()})
            agg.setdefault(metric, {})[m] = (mean, lo, hi)
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("model")
    summary.to_csv(os.path.join(CMP, "summary_table.csv"))
    print(summary.round(3).to_string())

    names = list(models)
    ev.plot_summary_bar(names, *zip(*(agg["pearson"][m] for m in names)),
                        os.path.join(CMP, "bar_pearson.png"),
                        "Held-out test: mean Pearson r over 20 perturbations (95% CI)",
                        ylabel="mean Pearson r (predicted vs observed log2FC)", ylim=(0, 1))
    ev.plot_summary_bar(names, *zip(*(agg["rmse"][m] for m in names)),
                        os.path.join(CMP, "bar_rmse.png"),
                        "Held-out test: mean RMSE over 20 perturbations (95% CI, lower is better)",
                        ylabel="mean RMSE (predicted vs observed log2FC)", ylim=(0, None))

    best = max(models, key=lambda m: summary.loc[m, "pearson_mean"])
    pair = {m: models[m]["metrics_test"] for m in dict.fromkeys(["baseline", best])}
    ev.plot_perturbation_ci(pair, os.path.join(CMP, "perturbation_ci.png"),
                            f"Per-perturbation Pearson r: baseline vs {best}  (bar = 95% bootstrap CI)",
                            metric="pearson")
    ev.plot_perturbation_ci(pair, os.path.join(CMP, "perturbation_ci_rmse.png"),
                            f"Per-perturbation RMSE: baseline vs {best}  (bar = 95% bootstrap CI)",
                            metric="rmse", xlabel="RMSE (predicted vs observed log2FC)")

    for m in dict.fromkeys([best, "baseline"]):
        ev.plot_scatter_panels(m, models[m]["pred_test"], targets.loc[split["test"]],
                               os.path.join(CMP, f"scatter_{m}.png"))

    json.dump({"best_model": best, "n_train": len(split["train"]), "n_test": len(split["test"])},
              open(os.path.join(CMP, "summary.json"), "w"), indent=2)
    print(f"\nwrote {CMP}/  (best model: {best})")


if __name__ == "__main__":
    main()
