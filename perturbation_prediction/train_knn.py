import argparse

import numpy as np

from harness import run_model


def _unit_rows(F):
    F = F - F.mean(axis=1, keepdims=True)        
    nrm = np.linalg.norm(F, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return F / nrm


class KNN:
    def __init__(self, ks=(1, 3, 5, 8, 12, 20), weight_power=2.0, use_ncells=True):
        self.ks = ks
        self.weight_power = weight_power
        self.use_ncells = use_ncells

    def _predict_from_sims(self, sims, k):
        pred = np.zeros((sims.shape[0], self.Y.shape[1]))
        for i in range(sims.shape[0]):
            order = np.argsort(-sims[i])[:k]
            w = np.clip(sims[i, order], 1e-6, None) ** self.weight_power
            if self.use_ncells:
                w = w * np.sqrt(self.n[order])
            w = w / w.sum()
            pred[i] = (w[:, None] * self.Y[order]).sum(0)
        return pred

    def fit(self, F, Y, n_cells):
        self.Fn = _unit_rows(F)
        self.Y = Y
        self.n = np.asarray(n_cells, dtype=float)

        S = self.Fn @ self.Fn.T
        np.fill_diagonal(S, -np.inf)               # leave-one-out: never pick self
        best_k, best_score = self.ks[0], -np.inf
        for k in self.ks:
            if k >= len(F):
                continue
            preds = self._predict_from_sims(S, k)
            r = np.mean([np.corrcoef(preds[i], Y[i])[0, 1] for i in range(len(F))])
            if r > best_score:
                best_score, best_k = r, k
        self.k = best_k
        self.loo_pearson = float(best_score)
        return self

    def predict(self, F):
        return self._predict_from_sims(_unit_rows(F) @ self.Fn.T, self.k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["raw", "scalars"], default="raw")
    args = ap.parse_args()
    run_model(f"knn_{args.features}", KNN, features=args.features)
