"""Closed-form weight solver from Theorem 3.1 (Chai & Wang, ICML 2022).

Given per-sample losses within a subgroup, hyperparameter alpha > 0, and
budget c, returns the optimal weights solving:

    max_w  sum_i  w_i * l_i  -  alpha * ||w||_2^2
    s.t.   sum_i w_i = c,   w_i >= 0.

The solution is a soft-threshold / water-filling rule.
"""

import numpy as np


def solve_subgroup_weights(losses: np.ndarray, alpha: float, c: float) -> np.ndarray:
    """Compute optimal per-sample weights for one subgroup.

    Parameters
    ----------
    losses : 1-D array of per-sample losses (non-negative).
    alpha  : regularisation strength (> 0).
    c      : total-weight budget for this subgroup.

    Returns
    -------
    w : 1-D array of weights, same length as *losses*.
    """
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    if n == 0:
        return np.zeros(0)

    order = np.argsort(-losses)
    sorted_l = losses[order]

    cum = 0.0
    lam = None
    for k in range(1, n + 1):
        cum += sorted_l[k - 1]
        cand = (cum - 2.0 * alpha * c) / k
        upper = sorted_l[k - 1]
        lower = sorted_l[k] if k < n else -np.inf
        if lower - 1e-12 <= cand <= upper + 1e-12:
            lam = cand
            break

    if lam is None:
        lam = (sorted_l.sum() - 2.0 * alpha * c) / n

    w = np.maximum((losses - lam) / (2.0 * alpha), 0.0)
    return w
