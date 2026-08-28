import argparse
import os
import pickle
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_prep import load_features, make_split
from evaluate_utils import (
    CLASSES,
    compute_metrics,
    cross_validate_report,
    majority_class_baseline,
    plot_confusion_matrix,
    plot_roc_curves,
    save_json,
)


def main(results_dir, max_iter, seed):
    os.makedirs(results_dir, exist_ok=True)

    print("Loading features...")
    X, y, _ = load_features()
    X_train, X_test, y_train, y_test, cv = make_split(X, y, seed=seed)
    print(f"train={X_train.shape}, test={X_test.shape}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(solver="lbfgs", max_iter=max_iter, random_state=seed)

    print("5-fold CV on training set...")
    cv_report = cross_validate_report(clf, X_train_s, y_train, cv)
    print(cv_report)

    print("Fitting on full training set...")
    t0 = time.time()
    clf.fit(X_train_s, y_train)
    fit_seconds = time.time() - t0

    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)
    # align predict_proba columns to CLASSES order
    proba_df = pd.DataFrame(y_proba, columns=clf.classes_)[CLASSES].to_numpy()

    metrics = compute_metrics(y_test, y_pred, proba_df)
    metrics["cv"] = cv_report
    metrics["baseline"] = majority_class_baseline(y_train, y_test)
    metrics["fit_seconds"] = fit_seconds
    metrics["model"] = "logistic_regression"
    save_json(metrics, os.path.join(results_dir, "metrics.json"))
    print(f"accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")

    plot_confusion_matrix(y_test, y_pred, CLASSES, os.path.join(results_dir, "confusion_matrix.png"),
                           title="Logistic regression")
    plot_roc_curves(y_test, proba_df, CLASSES, os.path.join(results_dir, "roc_curves.png"),
                     title="Logistic regression ROC")

    with open(os.path.join(results_dir, "model.pkl"), "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler}, f)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/logreg")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.results_dir, args.max_iter, args.seed)
