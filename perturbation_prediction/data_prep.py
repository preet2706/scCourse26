import json
import os

import numpy as np
import pandas as pd
import scanpy as sc

HERE = os.path.dirname(__file__)
RNA_PATH = os.path.join(HERE, "..", "rna_filtered.h5ad")
CLUSTER_LABELS = os.path.join(HERE, "..", "perturbation_clustering", "results", "cluster_labels.csv")
RESULTS = os.path.join(HERE, "results")

TARGETS_CSV = os.path.join(RESULTS, "log2fc_targets.csv")
FEATURES_NPZ = os.path.join(RESULTS, "gene_features.npz")
SPLIT_JSON = os.path.join(RESULTS, "split_perturbations.json")
CELLCACHE_NPZ = os.path.join(RESULTS, "cell_cache.npz")   # per-cell HVG counts for noise-ceiling / bootstrap

N_TRAIN = 50
N_TEST = 20
MIN_CELLS = 50          # min Control-condition cells for a perturbation to be selectable
N_PCS = 20              # PCs whose cell-scores the scalar features correlate each gene against
PRIOR = 0.3             # shrinkage prior count (per-1e4 units) in the pseudobulk log2 ratio
SEED = 0


def _load_control(adata_path=RNA_PATH):

    adata = sc.read_h5ad(adata_path)
    ctrl = adata[adata.obs["perturbation_2"] == "Control"].copy()
    del adata
    hvg = ctrl.var["highly_variable"].to_numpy()
    return ctrl, hvg


def build_targets(ctrl, hvg, perturbations, save=True):
    
    hvg_names = ctrl.var_names[hvg].to_numpy()
    X = ctrl[:, hvg].X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
    norm_counts = np.expm1(X)                       # back to library-normalised counts

    pert = ctrl.obs["perturbation"].to_numpy().astype(str)
    pseudobulk = {}
    for p in list(perturbations) + ["control"]:
        pseudobulk[p] = norm_counts[pert == p].mean(axis=0)
    pb = pd.DataFrame(pseudobulk, index=hvg_names).T   # rows = perturbations, cols = HVGs

    ctrl_vec = pb.loc["control"].to_numpy()
    log2fc = np.log2((pb.loc[list(perturbations)].to_numpy() + PRIOR) / (ctrl_vec + PRIOR))
    log2fc = pd.DataFrame(log2fc, index=list(perturbations), columns=hvg_names)

    if save:
        log2fc.to_csv(TARGETS_CSV)
        print(f"targets: {log2fc.shape[0]} perturbations x {log2fc.shape[1]} HVGs  "
              f"(prior={PRIOR}, |log2fc| mean={log2fc.abs().mean().mean():.3f}) -> {TARGETS_CSV}")
        # compact per-cell cache (selected perturbations + control) for split-half / bootstrap CIs
        keep = np.isin(pert, list(perturbations) + ["control"])
        np.savez(CELLCACHE_NPZ,
                 norm_counts=norm_counts[keep].astype(np.float32),
                 perturbation=pert[keep].astype(str),
                 hvg_names=hvg_names, prior=np.float64(PRIOR))
        print(f"cell cache: {int(keep.sum())} cells x {len(hvg_names)} HVGs -> {CELLCACHE_NPZ}")
    return log2fc



def build_gene_features(ctrl, hvg, target_genes, save=True):
    nt = ctrl[ctrl.obs["perturbation"].astype(str) == "control"]
    hvg_names = ctrl.var_names[hvg].to_numpy()

    present = [g for g in target_genes if g in set(ctrl.var_names)]
    missing = sorted(set(target_genes) - set(present))
    if missing:
        print(f"gene features: {len(missing)} target gene(s) absent from var_names, skipped: {missing}")

    Hs = nt[:, hvg].X
    Hs = np.asarray(Hs.todense() if hasattr(Hs, "todense") else Hs, dtype=np.float64)   # (cells x 2000) log1p
    G = nt[:, present].X
    G = np.asarray(G.todense() if hasattr(G, "todense") else G, dtype=np.float64)        # (cells x n_genes) log1p
    n_cells = Hs.shape[0]

    def zcols(M):
        mu = M.mean(0, keepdims=True)
        sd = M.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        return (M - mu) / sd

    Hz, Gz = zcols(Hs), zcols(G)
    coexp = (Gz.T @ Hz) / n_cells                      # (n_genes x 2000) Pearson corr

    # scalar block
    mean_expr = G.mean(0)
    detect_frac = (G > 0).mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.where(mean_expr > 0, G.var(0) / mean_expr, 0.0)

    sc.pp.pca(nt, n_comps=N_PCS, mask_var="highly_variable")
    pc_scores = nt.obsm["X_pca"][:, :N_PCS]            # (cells x N_PCS)
    pc_corr = (Gz.T @ zcols(pc_scores)) / n_cells      # (n_genes x N_PCS)

    scalars = np.column_stack([mean_expr, detect_frac, dispersion, pc_corr])
    scalar_names = ["mean_expr", "detect_frac", "dispersion"] + [f"pc{i+1}_corr" for i in range(N_PCS)]

    out = {
        "genes": np.array(present),
        "coexp": coexp.astype(np.float32),
        "scalars": scalars.astype(np.float32),
        "scalar_names": np.array(scalar_names),
        "hvg_names": hvg_names,
    }
    if save:
        np.savez(FEATURES_NPZ, **out)
        print(f"gene features: {coexp.shape[0]} genes x (coexp {coexp.shape[1]}, scalars {scalars.shape[1]}) -> {FEATURES_NPZ}")
    return out


def make_split(ctrl, save=True):
    
    if save and os.path.exists(SPLIT_JSON):
        with open(SPLIT_JSON) as f:
            return json.load(f)

    clusters = pd.read_csv(CLUSTER_LABELS).set_index("perturbation")["hierarchical"]
    counts = ctrl.obs["perturbation"].value_counts()
    counts = counts[counts.index.astype(str) != "control"]

    elig = pd.DataFrame({"n_cells": counts}).join(clusters, how="inner")
    elig = elig[elig["n_cells"] >= MIN_CELLS].dropna()
    # a perturbation is only usable if the knocked-out gene is measured (needs a feature vector)
    elig = elig[elig.index.isin(set(ctrl.var_names))]
    elig["hierarchical"] = elig["hierarchical"].astype(int)

    rng = np.random.default_rng(SEED)
    by_cluster = {c: sub.sort_values("n_cells", ascending=False).index.tolist()
                  for c, sub in elig.groupby("hierarchical")}
    sizes = {c: len(v) for c, v in by_cluster.items()}
    total = sum(sizes.values())

    def allocate(n_total):
        raw = {c: n_total * s / total for c, s in sizes.items()}
        alloc = {c: int(np.floor(v)) for c, v in raw.items()}
        while sum(alloc.values()) < n_total:
            c = max(raw, key=lambda k: raw[k] - alloc[k])
            alloc[c] += 1
        return alloc

    a_train, a_test = allocate(N_TRAIN), allocate(N_TEST)
    train, test = [], []
    leftover_need_train = leftover_need_test = 0
    remaining = {}
    for c, genes in by_cluster.items():
        nt_ = min(a_train[c], len(genes))
        train += genes[:nt_]
        rest = genes[nt_:]
        ne_ = min(a_test[c], len(rest))
        test += rest[:ne_]
        remaining[c] = rest[ne_:]
        leftover_need_train += a_train[c] - nt_
        leftover_need_test += a_test[c] - ne_

    pool = [g for c in remaining for g in remaining[c]]
    rng.shuffle(pool)
    train += pool[:leftover_need_train]
    test += pool[leftover_need_train:leftover_need_train + leftover_need_test]

    assert not (set(train) & set(test)), "train/test overlap"
    split = {
        "train": sorted(train),
        "test": sorted(test),
        "cluster_of": {g: int(elig.loc[g, "hierarchical"]) for g in train + test},
        "n_cells": {g: int(elig.loc[g, "n_cells"]) for g in train + test},
    }
    if save:
        with open(SPLIT_JSON, "w") as f:
            json.dump(split, f, indent=2)
        tr = pd.Series([split["cluster_of"][g] for g in split["train"]]).value_counts().sort_index()
        te = pd.Series([split["cluster_of"][g] for g in split["test"]]).value_counts().sort_index()
        print(f"split: {len(train)} train + {len(test)} test -> {SPLIT_JSON}")
        print(f"  train per hierarchical cluster: {tr.to_dict()}")
        print(f"  test  per hierarchical cluster: {te.to_dict()}")
    return split


def load_split():
    with open(SPLIT_JSON) as f:
        return json.load(f)


def load_targets():
    return pd.read_csv(TARGETS_CSV, index_col=0)


def load_cell_cache():
    d = np.load(CELLCACHE_NPZ, allow_pickle=True)
    return d["norm_counts"], d["perturbation"].astype(str), d["hvg_names"], float(d["prior"])


def load_gene_features(which="raw"):

    d = np.load(FEATURES_NPZ, allow_pickle=True)
    genes = d["genes"]
    coexp = d["coexp"].astype(np.float64)
    if which == "raw":
        F = coexp
        names = [f"coexp:{g}" for g in d["hvg_names"]]
    elif which == "scalars":
        sc_ = d["scalars"].astype(np.float64)
        sc_z = (sc_ - sc_.mean(0)) / (sc_.std(0) + 1e-8)
        F = np.column_stack([coexp, sc_z])
        names = [f"coexp:{g}" for g in d["hvg_names"]] + [f"scalar:{n}" for n in d["scalar_names"]]
    else:
        raise ValueError(which)
    return pd.DataFrame(F, index=genes, columns=names), d["hvg_names"]


def build_all(force=False):
    
    os.makedirs(RESULTS, exist_ok=True)
    ctrl, hvg = _load_control()
    print(f"Control condition: {ctrl.n_obs} cells, {int(hvg.sum())} HVGs, "
          f"{ctrl.obs['perturbation'].nunique()} perturbation categories")

    if force and os.path.exists(SPLIT_JSON):
        os.remove(SPLIT_JSON)
    split = make_split(ctrl)

    perturbations = split["train"] + split["test"]
    if force or not os.path.exists(TARGETS_CSV):
        build_targets(ctrl, hvg, perturbations)
    if force or not os.path.exists(FEATURES_NPZ):
        all_targets = pd.read_csv(CLUSTER_LABELS)["perturbation"].tolist()
        build_gene_features(ctrl, hvg, all_targets)

    tgt = load_targets()
    on_target = [tgt.loc[p, p] for p in perturbations if p in tgt.columns]
    print(f"on-target log2FC: median={np.median(on_target):.3f}, "
          f"{np.mean(np.array(on_target) < 0)*100:.0f}% negative "
          f"({len(on_target)} of {len(perturbations)} selected genes are themselves HVGs)")
    return split


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild cached artefacts")
    build_all(force=ap.parse_args().force)
