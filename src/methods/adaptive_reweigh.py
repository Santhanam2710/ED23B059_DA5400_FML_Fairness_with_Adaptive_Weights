"""Algorithm 1 — Iterative Adaptive Reweighing (Chai & Wang, ICML 2022).

Pretrain with uniform weights, then iterate:
    1. Compute per-sample cross-entropy losses.
    2. Solve closed-form weights per subgroup (Theorem 3.1).
    3. Refit the base classifier with new sample weights.
Stop when weights converge or max iterations reached.
"""

import numpy as np
from .weight_solver import solve_subgroup_weights


def _per_sample_logloss(clf, X, y):
    """Binary cross-entropy per sample."""
    p = clf.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def adaptive_reweigh_fit(X, y, s, base_factory, alpha=10.0,
                         max_iter=15, tol=1e-3):
    """Train a classifier with adaptive reweighing.

    Parameters
    ----------
    X, y, s     : training arrays.
    base_factory: callable returning a fresh sklearn classifier that
                  supports ``sample_weight`` in ``fit()``.
    alpha       : Eq. (3) regularisation parameter.
    max_iter    : maximum reweighing iterations.
    tol         : relative L2 convergence threshold on weights.

    Returns
    -------
    clf         : trained classifier.
    w           : final sample weights.
    history     : list of dicts with per-iteration diagnostics.
    """
    n = len(X)
    y = y.astype(int)
    s = s.astype(int)

    # Subgroup indices: (label, sensitive)
    subgroup_idx = {}
    for yy in [0, 1]:
        for ss in [0, 1]:
            mask = (y == yy) & (s == ss)
            if mask.sum() > 0:
                subgroup_idx[(yy, ss)] = np.where(mask)[0]

    # c = size of largest subgroup (paper's choice)
    c = max(len(ix) for ix in subgroup_idx.values())

    # Pre-train with uniform weights
    w = np.ones(n)
    clf = base_factory()
    clf.fit(X, y, sample_weight=w)

    history = []
    for it in range(max_iter):
        losses = _per_sample_logloss(clf, X, y)
        new_w = np.zeros(n)
        for key, idx in subgroup_idx.items():
            new_w[idx] = solve_subgroup_weights(losses[idx], alpha=alpha, c=c)
        if new_w.sum() < 1e-9:
            new_w = np.ones(n)

        clf = base_factory()
        clf.fit(X, y, sample_weight=new_w)

        delta = np.linalg.norm(new_w - w) / (np.linalg.norm(w) + 1e-9)
        history.append({"iter": it, "delta": float(delta),
                        "nonzero_frac": float((new_w > 0).mean())})
        if delta < tol:
            break
        w = new_w

    return clf, new_w, history
