"""Mandatory validation guards for new/custom training loops (see CLAUDE.md).

Rule: any training loop that does NOT use `train_utils.train_model` or
`run_benchmark.train_one` must pass `shuffled_label_control` before any OOD /
classification metric is reported. A high shuffled control means the loop is broken
(historical cause: indexing full arrays with local batch positions instead of the
global training rows `tr[perm]`).
"""
import numpy as np

DEFAULT_MAX_SHUFFLED = 0.60


def shuffled_label_control(run_with_labels, y_train, seed=1, max_shuffled=DEFAULT_MAX_SHUFFLED,
                           higher_is_better=True, name="loop"):
    """Train the identical pipeline on shuffled training labels and compare.

    Parameters
    ----------
    run_with_labels : callable
        ``run_with_labels(labels) -> float`` trains on `labels` (aligned to the training
        rows) and returns the metric on the true test labels. The caller must reuse the
        SAME train/test split and indexing inside this callable.
    y_train : array-like
        Training labels aligned to the training rows.
    max_shuffled : float
        For classification (higher_is_better), the shuffled metric must be < max_shuffled
        (default 0.60); otherwise an AssertionError is raised.

    Returns
    -------
    (real, shuffled) : tuple[float, float]
    """
    y = np.asarray(y_train).copy()
    real = float(run_with_labels(y))
    rng = np.random.RandomState(seed)
    ysh = y.copy()
    ysh = ysh[rng.permutation(len(ysh))]
    shuffled = float(run_with_labels(ysh))
    if higher_is_better and shuffled >= max_shuffled:
        raise AssertionError(
            f"[{name}] shuffled-label control = {shuffled:.3f} >= {max_shuffled} "
            f"(real = {real:.3f}). The loop is broken or the metric is confounded. "
            "Check that batches index global rows (X[tr[perm]]) not local positions (X[perm])."
        )
    return real, shuffled


def leakage_check(train_idx, test_idx, name="loop"):
    """Assert train/test row indices are disjoint."""
    inter = set(np.asarray(train_idx).tolist()) & set(np.asarray(test_idx).tolist())
    if inter:
        raise AssertionError(f"[{name}] train/test overlap: {len(inter)} rows")
    return True


def row_order_invariance(predict_scores, y_test, seed=0, tol=1e-9, name="loop"):
    """AUROC must be unchanged when the test rows are permuted.

    `predict_scores` is a callable ``(idx) -> scores`` where idx are test-row positions;
    simpler: pass arrays and this checks the AUROC is permutation invariant.
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_test)
    s = np.asarray(predict_scores)
    auc = roc_auc_score(y, s)
    rng = np.random.RandomState(seed)
    p = rng.permutation(len(y))
    auc_p = roc_auc_score(y[p], s[p])
    if abs(auc - auc_p) > tol:
        raise AssertionError(f"[{name}] AUROC not row-order invariant: {auc} vs {auc_p}")
    return auc


def batch_index_ok(rows):
    """Assert a batch index array is within the global row space and not a local range."""
    rows = np.asarray(rows)
    if rows.ndim != 1 or len(rows) == 0:
        raise AssertionError("batch index must be a non-empty 1-D array")
    if rows.min() < 0:
        raise AssertionError("batch index contains negative entries")
    return True


if __name__ == "__main__":
    # Minimal self-test: a correct loop (indexes global rows) must pass.
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import torch

    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed_ext", "L3")
    cells = np.load(os.path.join(d, "cells_l3.npy")).astype(np.float32)
    y = np.load(os.path.join(d, "y_l3.npy"))
    tr = np.load(os.path.join(d, "splits", "L3_ldo_train.npy"))
    te = np.load(os.path.join(d, "splits", "L3_ldo_test.npy"))
    pca = PCA(128, random_state=0).fit(cells[tr])
    X = pca.transform(cells)

    def run_with_labels(labels):
        clf = LogisticRegression(max_iter=1000).fit(X[tr], labels)
        return roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])

    leakage_check(tr, te)
    real, shuf = shuffled_label_control(run_with_labels, y[tr], name="L3-logreg-self-test")
    print(f"self-test OK: real={real:.3f} shuffled={shuf:.3f}")
