import os

import numpy as np
import anndata as ad
from sklearn.model_selection import train_test_split, StratifiedKFold

DEFAULT_H5AD = os.path.join(os.path.dirname(__file__), "..", "rna_filtered.h5ad")
SPLIT_PATH = os.path.join(os.path.dirname(__file__), "results", "split_indices.npz")


def load_features(h5ad_path: str = DEFAULT_H5AD):
    
    adata = ad.read_h5ad(h5ad_path)
    hvg_mask = adata.var["highly_variable"].to_numpy()
    gene_names = adata.var_names[hvg_mask].to_numpy()

    X = adata[:, hvg_mask].X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    y = adata.obs['perturbation_2'].astype(str).to_numpy()

    return X, y, gene_names


def make_split(X, y, test_size: float = 0.2, seed: int = 0, n_cv_folds: int = 5,
               save: bool = True, split_path: str = SPLIT_PATH):

    n = X.shape[0]
    idx = np.arange(n)

    if save and os.path.exists(split_path):
        cached = np.load(split_path)
        train_idx, test_idx = cached["train_idx"], cached["test_idx"]
    else:
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=y
        )
        if save:
            os.makedirs(os.path.dirname(split_path), exist_ok=True)
            np.savez(split_path, train_idx=train_idx, test_idx=test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)

    return X_train, X_test, y_train, y_test, cv
