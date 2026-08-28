import argparse
import os

import numpy as np
from sklearn.decomposition import PCA

from harness import run_model


class MLPEnsemble:
    def __init__(self, n_models=5, hidden=(64, 32), dropout=0.5, weight_decay=1e-3, lr=3e-3,
                 epochs=300, latent=20, seed=0, n_threads=1):
        self.cfg = dict(n_models=n_models, hidden=hidden, dropout=dropout, weight_decay=weight_decay,
                        lr=lr, epochs=epochs, latent=latent, seed=seed, n_threads=n_threads)

    def _make_net(self, d_in, d_out, torch):
        layers, prev = [], d_in
        for h in self.cfg["hidden"]:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(self.cfg["dropout"])]
            prev = h
        layers.append(torch.nn.Linear(prev, d_out))
        return torch.nn.Sequential(*layers)

    def fit(self, F, Y, n_cells):
        import torch
        torch.set_num_threads(self.cfg["n_threads"])
        self.torch = torch

        self.mu, self.sd = F.mean(0), F.std(0) + 1e-8
        Xt = torch.tensor((F - self.mu) / self.sd, dtype=torch.float32)

        k = min(self.cfg["latent"], Y.shape[0] - 1)
        self.pca = PCA(n_components=k, random_state=0).fit(Y)
        Z = self.pca.transform(Y)
        self.z_mu, self.z_sd = Z.mean(0), Z.std(0) + 1e-8
        Zt = torch.tensor((Z - self.z_mu) / self.z_sd, dtype=torch.float32)

        self.nets = []
        for m in range(self.cfg["n_models"]):
            torch.manual_seed(self.cfg["seed"] + m)
            net = self._make_net(F.shape[1], k, torch)
            opt = torch.optim.Adam(net.parameters(), lr=self.cfg["lr"],
                                   weight_decay=self.cfg["weight_decay"])
            loss_fn = torch.nn.MSELoss()
            net.train()
            for _ in range(self.cfg["epochs"]):
                opt.zero_grad()
                loss_fn(net(Xt), Zt).backward()
                opt.step()
            net.eval()
            self.nets.append(net)
        return self

    def predict(self, F):
        torch = self.torch
        Xt = torch.tensor((F - self.mu) / self.sd, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for net in self.nets:
                z = net(Xt).numpy() * self.z_sd + self.z_mu
                preds.append(self.pca.inverse_transform(z))
        return np.mean(preds, axis=0)


def make_factory(args):
    n_threads = args.n_jobs if args.n_jobs and args.n_jobs > 0 else (os.cpu_count() or 1)
    return lambda: MLPEnsemble(n_models=args.n_models, epochs=args.epochs, latent=args.latent,
                               n_threads=n_threads)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["raw", "scalars"], default="raw")
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--latent", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()
    run_model(f"mlp_{args.features}", make_factory(args), features=args.features)
