"""Fairness and accuracy evaluation metrics.

All gap metrics follow the paper's definitions (Section 3.1).
Lower gap = fairer.
"""

import numpy as np
from sklearn.metrics import accuracy_score


def fairness_metrics(y_true, y_pred, s):
    """Compute accuracy and three fairness gap metrics.

    Parameters
    ----------
    y_true, y_pred, s : array-like of shape (n,).

    Returns
    -------
    dict with keys: acc, disp_impact, disp_tpr, disp_tnr.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    s = np.asarray(s, dtype=int)

    def rate(mask):
        return y_pred[mask].mean() if mask.sum() else 0.0

    def tpr(mask):
        m = mask & (y_true == 1)
        return y_pred[m].mean() if m.sum() else 0.0

    def tnr(mask):
        m = mask & (y_true == 0)
        return (1 - y_pred[m]).mean() if m.sum() else 0.0

    s0, s1 = (s == 0), (s == 1)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "disp_impact": float(abs(rate(s0) - rate(s1))),
        "disp_tpr": float(abs(tpr(s0) - tpr(s1))),
        "disp_tnr": float(abs(tnr(s0) - tnr(s1))),
    }
