import os

import numpy as np
import pandas as pd

import data_prep as dp
import evaluate_utils as ev

RESULTS = dp.RESULTS


def run_model(name, make_model, features="raw", n_boot=1000):
    outdir = os.path.join(RESULTS, name)
    os.makedirs(outdir, exist_ok=True)

    split = dp.load_split()
    targets = dp.load_targets()
    hvg_names = targets.columns.to_numpy()
    F, _ = dp.load_gene_features(which=features)

    train = [p for p in split["train"] if p in F.index]
    test = [p for p in split["test"] if p in F.index]
    ncell = split["n_cells"]

    Ftr, Ytr = F.loc[train].to_numpy(), targets.loc[train].to_numpy()
    ntr = np.array([ncell[p] for p in train], dtype=float)

    model = make_model()
    model.fit(Ftr, Ytr, ntr)
    pred = pd.DataFrame(model.predict(F.loc[test].to_numpy()), index=test, columns=hvg_names)

    counts, pertarr, _, prior = dp.load_cell_cache()
    rows = []
    for p in test:
        obs = targets.loc[p].to_numpy()
        pv = pred.loc[p].to_numpy()
        (pm, plo, phi), (rm, rlo, rhi) = ev.bootstrap_ci(counts, pertarr, prior, p, pv, n_boot=n_boot)
        rows.append({"perturbation": p, "n_cells": int(ncell[p]),
                     "pearson": ev.pearson(pv, obs),
                     "pearson_boot_mean": pm, "pearson_ci_lo": plo, "pearson_ci_hi": phi,
                     "rmse": ev.rmse(pv, obs),
                     "rmse_boot_mean": rm, "rmse_ci_lo": rlo, "rmse_ci_hi": rhi})
    metrics = pd.DataFrame(rows).set_index("perturbation")

    pred.to_csv(os.path.join(outdir, "predictions_test.csv"))
    metrics.to_csv(os.path.join(outdir, "metrics_test.csv"))
    summary = {"model": name, "features": features, "n_train": len(train), "n_test": len(test)}
    for m in ("pearson", "rmse"):
        summary[f"{m}_mean"] = float(metrics[m].mean())
        summary[f"{m}_sem"] = float(metrics[m].sem())
        summary[f"{m}_median"] = float(metrics[m].median())
        summary[f"{m}_min"] = float(metrics[m].min())
        summary[f"{m}_max"] = float(metrics[m].max())
    ev.save_json(summary, os.path.join(outdir, "metrics.json"))
    print(f"[{name}] test  Pearson mean={summary['pearson_mean']:.3f} median={summary['pearson_median']:.3f}"
          f"   RMSE mean={summary['rmse_mean']:.3f} median={summary['rmse_median']:.3f}")
    return summary
