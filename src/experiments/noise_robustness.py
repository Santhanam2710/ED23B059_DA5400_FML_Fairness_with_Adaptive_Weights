"""Experiment 3 — Noise robustness analysis.

Evaluates all 4 methods under group-dependent label noise at varying
ratios.  Reproduces the direction of the paper's Figure 3.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.datasets import load_adult, load_german
from src.helpers import fairness_metrics
from src.methods import lr_factory, rf_factory, svm_factory, adaptive_reweigh_fit


def _inject_noise(y, noise_ratio, s, seed=0):
    """Group-dependent noise: flip labels independently per group."""
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    for group in [0, 1]:
        group_idx = np.where(s == group)[0]
        n_flip = int(len(group_idx) * noise_ratio)
        if n_flip > 0:
            flip_idx = rng.choice(group_idx, size=n_flip, replace=False)
            y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return y_noisy


def _run_single(X_tr, y_tr, s_tr, X_te, y_te, s_te, method, alpha=10.0):
    if method == "LR":
        clf = lr_factory(); clf.fit(X_tr, y_tr)
    elif method == "RF":
        clf = rf_factory(); clf.fit(X_tr, y_tr)
    elif method == "SVM":
        clf = svm_factory(); clf.fit(X_tr, y_tr)
    elif method == "LR+Adaptive":
        clf, _, _ = adaptive_reweigh_fit(
            X_tr, y_tr, s_tr, base_factory=lr_factory, alpha=alpha)
    else:
        raise ValueError(method)
    y_hat = clf.predict(X_te)
    return fairness_metrics(y_te, y_hat, s_te)


def run(seeds, test_size, noise_ratios, results_dir):
    """Run noise robustness experiment on both datasets."""
    datasets = [("Adult", load_adult), ("German", load_german)]
    methods = ["LR", "RF", "SVM", "LR+Adaptive"]
    rows = []

    for ds_name, loader in datasets:
        X, y, s = loader()
        for noise in noise_ratios:
            for seed in seeds:
                X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                    X, y, s, test_size=test_size, random_state=seed, stratify=y)
                if noise > 0:
                    y_tr = _inject_noise(y_tr, noise, s_tr, seed=seed)
                for m in methods:
                    metrics = _run_single(
                        X_tr, y_tr, s_tr, X_te, y_te, s_te, m, alpha=10.0)
                    metrics.update({"noise": noise, "method": m,
                                    "seed": seed, "dataset": ds_name})
                    rows.append(metrics)
            print(f"  {ds_name} | noise={noise:.0%} done")

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/tables/noise_robustness_results.csv", index=False)

    _plot_noise(df, results_dir)
    return df


_METHOD_STYLES = {
    "LR":          dict(color="steelblue",   marker="o", ls="-",  label="LR (Baseline)"),
    "RF":          dict(color="darkorange",  marker="s", ls="--", label="RF"),
    "SVM":         dict(color="forestgreen", marker="^", ls=":",  label="SVM"),
    "LR+Adaptive": dict(color="crimson",     marker="D", ls="-",  label="LR+Adaptive (Ours)"),
}

_METRICS_PLOT = [
    ("acc",         "Classification Accuracy"),
    ("disp_impact", "Disparate Impact"),
    ("disp_tpr",    "Disparate TPR"),
    ("disp_tnr",    "Disparate TNR"),
]


def _plot_noise(df, results_dir):
    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    for col, dataset in enumerate(["Adult", "German"]):
        sub = df[df["dataset"] == dataset]
        for row, (metric, label) in enumerate(_METRICS_PLOT):
            ax = axes[row, col]
            for m, style in _METHOD_STYLES.items():
                d = sub[sub["method"] == m]
                g = d.groupby("noise")[metric].agg(["mean", "std"])
                ax.errorbar(g.index * 100, g["mean"], yerr=g["std"],
                            marker=style["marker"], color=style["color"],
                            linestyle=style["ls"], label=style["label"],
                            capsize=3, linewidth=2, markersize=5)
            ax.set_xlabel("Noise Ratio (%)", fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(f"{label}: {dataset}", fontsize=10, weight="bold")
            ax.legend(fontsize=7.5, loc="best")
            ax.grid(alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    plt.tight_layout(h_pad=2.5)
    plt.savefig(
        f"{results_dir}/figures/Change_of_accuracy_and_fairness_under_different_noise_ratio_on_Adult_and_German_datasets.png",
        dpi=150, bbox_inches="tight")
    plt.close()
