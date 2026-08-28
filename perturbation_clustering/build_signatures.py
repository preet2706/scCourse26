import numpy as np
import pandas as pd
import scanpy as sc

RNA_PATH = "/scratch/shahp/scCourse26/rna_filtered.h5ad"
MIN_CELLS = 15
OUT_SIGNATURES = "results/signatures.csv"
OUT_CELL_COUNTS = "results/perturbation_cell_counts.csv"


def main():
    adata = sc.read_h5ad(RNA_PATH)
    ctrl = adata[adata.obs["perturbation_2"] == "Control"].copy()
    hvg = ctrl.var_names[ctrl.var["highly_variable"]]
    ctrl = ctrl[:, hvg].copy()

    counts = ctrl.obs["perturbation"].value_counts()
    counts.name = "n_cells"
    counts.to_csv(OUT_CELL_COUNTS)

    keep_genes = counts[counts >= MIN_CELLS].index.tolist()
    dropped = counts[counts < MIN_CELLS]
    print(f"Control condition: {ctrl.n_obs} cells, {len(counts)} perturbations")
    print(f"Keeping {len(keep_genes)} perturbations with >= {MIN_CELLS} cells")
    print(f"Dropped ({len(dropped)}): {dropped.to_dict()}")

    X = ctrl.X.toarray() if not isinstance(ctrl.X, np.ndarray) else ctrl.X
    expr = pd.DataFrame(X, index=ctrl.obs_names, columns=hvg)
    expr["perturbation"] = ctrl.obs["perturbation"].values

    pseudobulk = expr.groupby("perturbation", observed=True).mean()
    control_mean = pseudobulk.loc["control"]

    keep_ko = [g for g in keep_genes if g != "control"]
    signatures = pseudobulk.loc[keep_ko] - control_mean

    signatures.to_csv(OUT_SIGNATURES)
    print(f"Saved {signatures.shape[0]} x {signatures.shape[1]} signature matrix -> {OUT_SIGNATURES}")


if __name__ == "__main__":
    main()
