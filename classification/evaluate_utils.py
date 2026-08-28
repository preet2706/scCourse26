import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize

CLASSES = ["Control", "IFNγ", "Co-culture"]


def majority_class_baseline(y_train, y_test):
    
    majority = pd.Series(y_train).value_counts().idxmax()
    y_pred = np.full_like(y_test, majority)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def compute_metrics(y_true, y_pred, y_proba, classes=CLASSES):
    y_true_bin = label_binarize(y_true, classes=classes)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_roc_auc_ovr": roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr"),
        "classification_report": classification_report(
            y_true, y_pred, labels=classes, output_dict=True, zero_division=0
        ),
    }
    return metrics


def cross_validate_report(estimator, X_train, y_train, cv, scoring="f1_macro"):
    scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    return {"cv_macro_f1_mean": float(scores.mean()), "cv_macro_f1_std": float(scores.std()),
            "cv_macro_f1_scores": scores.tolist()}


def plot_confusion_matrix(y_true, y_pred, classes, out_path, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, mat, fmt, t in zip(axes, [cm, cm_norm], ["d", ".2f"], ["counts", "row-normalized"]):
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} ({t})")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true, y_proba, classes, out_path, title="ROC curves (one-vs-rest)"):
    y_true_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curve(history, out_path, title="Training curve"):
    """history: list of {"epoch": int, "train_loss": float, "val_loss": float} (as saved by train_nn.py).
    Marks the best (lowest val-loss) epoch, which is where early stopping restored the model from."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    best_epoch = int(np.argmin(val_loss))

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(epochs, train_loss, color="#0072B2", lw=2, label="training loss")
    ax.plot(epochs, val_loss, color="#D55E00", lw=2, label="validation loss")
    ax.axvline(epochs[best_epoch], color="#999999", ls="--", lw=1)
    ax.annotate("best epoch", (epochs[best_epoch], val_loss[best_epoch]),
                xytext=(8, 8), textcoords="offset points", fontsize=9, color="#555555")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class SklearnLikeWrapper(ClassifierMixin, BaseEstimator):

    def __init__(self, predict_proba_fn, classes):
        self._predict_proba_fn = predict_proba_fn
        self.classes_ = np.array(classes)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self._predict_proba_fn(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def permutation_importance_for(model, X_test, y_test, gene_names, n_repeats=10, seed=0, n_jobs=1):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=seed,
        scoring="f1_macro", n_jobs=n_jobs,
    )
    return pd.DataFrame({
        "gene": gene_names,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
