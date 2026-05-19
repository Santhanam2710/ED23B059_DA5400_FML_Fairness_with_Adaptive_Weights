"""Experiment 2 — Alpha sensitivity sweep.

Varies alpha for LR+Adaptive and tracks accuracy + Disparate FPR.
Reproduces the direction of the paper's Figure 4.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.datasets import load_adult, load_german
from src.helpers import fairness_metrics
from src.methods import lr_factory, adaptive_reweigh_fit


def _run_single_adaptive(X_tr, y_tr, s_tr, X_te, y_te, s_te, alpha):
    clf, _, _ = adaptive_reweigh_fit(
        X_tr, y_tr, s_tr, base_factory=lr_factory, alpha=alpha)
    y_hat = clf.predict(X_te)
    return fairness_metrics(y_te, y_hat, s_te)


def run(seeds, test_size, alpha_grid, results_dir):
    """Run the alpha-sensitivity experiment."""
    datasets = [("Adult", load_adult), ("German", load_german)]
    rows = []

    for ds_name, loader in datasets:
        X, y, s = loader()
        for seed in seeds:
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, s, test_size=test_size, random_state=seed, stratify=y)
            for a in alpha_grid:
                m = _run_single_adaptive(X_tr, y_tr, s_tr, X_te, y_te, s_te, a)
                m.update({"alpha": a, "seed": seed, "dataset": ds_name})
                rows.append(m)
        print(f"  {ds_name} alpha sweep done")

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/tables/alpha_sweep_results.csv", index=False)

    _plot_alpha(df, results_dir)
    return df


def _plot_alpha(df, results_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, name in enumerate(["Adult", "German"]):
        sweep = df[df["dataset"] == name]
        g = sweep.groupby("alpha").agg(
            acc_mean=("acc", "mean"),      acc_std=("acc", "std"),
            tnr_mean=("disp_tnr", "mean"), tnr_std=("disp_tnr", "std"),
        )
        ax = axes[0, col]
        ax.errorbar(g.index, g["acc_mean"], yerr=g["acc_std"],
                    marker="o", color="steelblue", capsize=4, linewidth=2, markersize=6)
        ax.set_xscale("log"); ax.set_title(f"{name} — Accuracy vs α", weight="bold")
        ax.set_xlabel("Values of α"); ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        ax = axes[1, col]
        ax.errorbar(g.index, g["tnr_mean"], yerr=g["tnr_std"],
                    marker="s", color="crimson", capsize=4, linewidth=2, markersize=6)
        ax.set_xscale("log"); ax.set_title(f"{name} — Disparate FPR vs α", weight="bold")
        ax.set_xlabel("Values of α"); ax.set_ylabel("Disparate FPR")
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/figures/Alpha_Sensitivity_Plot.png",
                dpi=150, bbox_inches="tight")
    plt.close()
