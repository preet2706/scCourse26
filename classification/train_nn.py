import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from data_prep import load_features, make_split
from evaluate_utils import (
    CLASSES,
    SklearnLikeWrapper,
    compute_metrics,
    majority_class_baseline,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_roc_curves,
    permutation_importance_for,
    save_json,
)


class MLP:

    def __init__(self, n_features, n_classes, hidden=(256, 64), dropout=0.2, lr=1e-3,
                 max_epochs=100, patience=8, batch_size=1024, seed=0, n_threads=None):
        import torch

        self.torch = torch
        torch.manual_seed(seed)
        if n_threads:
            torch.set_num_threads(n_threads)

        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, n_classes))
        self.net = torch.nn.Sequential(*layers)

        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.history = []

    def fit(self, X_train, y_train_int, X_val, y_val_int):
        torch = self.torch
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_int, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_int, dtype=torch.long)

        n = X_train_t.shape[0]
        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = X_train_t[idx], y_train_t[idx]
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * self.batch_size
            train_loss = total_loss / n

            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(X_val_t), y_val_t).item()
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()


def main(results_dir, n_jobs, max_epochs, seed):
    os.makedirs(results_dir, exist_ok=True)

    print("Loading features...")
    X, y, gene_names = load_features()
    X_train, X_test, y_train, y_test, cv = make_split(X, y, seed=seed)
    print(f"train={X_train.shape}, test={X_test.shape}")

    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_s, y_train, test_size=0.15, random_state=seed, stratify=y_train
    )

    label_to_int = {c: i for i, c in enumerate(CLASSES)}
    int_to_label = np.array(CLASSES)
    y_tr_int = np.array([label_to_int[v] for v in y_tr])
    y_val_int = np.array([label_to_int[v] for v in y_val])

    n_threads = n_jobs if n_jobs and n_jobs > 0 else os.cpu_count()
    model = MLP(n_features=X.shape[1], n_classes=len(CLASSES), max_epochs=max_epochs,
                seed=seed, n_threads=n_threads)

    print("Training...")
    t0 = time.time()
    model.fit(X_tr, y_tr_int, X_val, y_val_int)
    fit_seconds = time.time() - t0

    y_proba = model.predict_proba(X_test_s)
    y_pred = int_to_label[np.argmax(y_proba, axis=1)]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["baseline"] = majority_class_baseline(y_train, y_test)
    metrics["fit_seconds"] = fit_seconds
    metrics["training_history"] = model.history
    metrics["model"] = "neural_network"
    save_json(metrics, os.path.join(results_dir, "metrics.json"))
    print(f"accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")

    plot_confusion_matrix(y_test, y_pred, CLASSES, os.path.join(results_dir, "confusion_matrix.png"),
                           title="Neural network")
    plot_roc_curves(y_test, y_proba, CLASSES, os.path.join(results_dir, "roc_curves.png"),
                     title="Neural network ROC")
    plot_loss_curve(model.history, os.path.join(results_dir, "loss_curve.png"),
                     title="Neural network training curve")
    
    print("Computing permutation importance on test set...")
    wrapped = SklearnLikeWrapper(model.predict_proba, CLASSES)
    perm_df = permutation_importance_for(wrapped, X_test_s, y_test, gene_names, n_jobs=1)
    perm_df.to_csv(os.path.join(results_dir, "importances.csv"), index=False)

    with open(os.path.join(results_dir, "model.pkl"), "wb") as f:
        pickle.dump({"state_dict": model.net.state_dict(), "mu": mu, "sigma": sigma,
                     "gene_names": gene_names, "classes": CLASSES}, f)

    print("Done. Top 15 genes by permutation importance:")
    print(perm_df.head(15))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/nn")
    parser.add_argument("--n-jobs", type=int, default=-1, help="torch CPU threads; -1 = all available")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.results_dir, args.n_jobs, args.max_epochs, args.seed)
