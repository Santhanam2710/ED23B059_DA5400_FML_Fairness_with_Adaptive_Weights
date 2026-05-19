"""Experiment 1 — Main 4-method comparison on Adult and German Credit.

Trains LR, RF, SVM (baselines) and LR+Adaptive across multiple seeds,
produces a summary table and accuracy-vs-fairness scatter + bar chart.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.datasets import load_adult, load_german
from src.methods import lr_factory, rf_factory, svm_factory, adaptive_reweigh_fit
from src.helpers import fairness_metrics


def _run_single(X_tr, y_tr, s_tr, X_te, y_te, s_te, method, alpha=10.0):
    """Train one method, return metrics dict."""
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


def run(seeds, test_size, alpha_adult, alpha_german, results_dir):
    """Run the main comparison experiment."""
    datasets = [
        ("Adult",  load_adult,  alpha_adult),
        ("German", load_german, alpha_german),
    ]
    methods = ["LR", "RF", "SVM", "LR+Adaptive"]
    rows = []

    for ds_name, loader, alpha in datasets:
        X, y, s = loader()
        for seed in seeds:
            X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
                X, y, s, test_size=test_size, random_state=seed, stratify=y)
            for m in methods:
                metrics = _run_single(
                    X_tr, y_tr, s_tr, X_te, y_te, s_te, m, alpha=alpha)
                metrics.update({"dataset": ds_name, "method": m, "seed": seed})
                rows.append(metrics)
                print(f"  {ds_name} | seed={seed} | {m:14s} "
                      f"acc={metrics['acc']:.4f}  DI={metrics['disp_impact']:.4f}  "
                      f"dTPR={metrics['disp_tpr']:.4f}  dTNR={metrics['disp_tnr']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/tables/raw_results.csv", index=False)

    # Summary table (mean ± std)
    summary = _summarise(df)
    summary.to_csv(f"{results_dir}/tables/summary_table.csv", index=False)
    print("\n" + summary.to_string(index=False))

    # Plots
    _plot_scatter(df, results_dir)
    _plot_fairness_bars(df, results_dir)

    return df


def _summarise(df):
    g = df.groupby(["dataset", "method"])
    mean = g[["acc", "disp_impact", "disp_tpr", "disp_tnr"]].mean()
    std  = g[["acc", "disp_impact", "disp_tpr", "disp_tnr"]].std()
    out = mean.copy()
    for col in mean.columns:
        out[col] = mean[col].round(4).astype(str) + " ± " + std[col].round(4).astype(str)
    order = ["LR", "RF", "SVM", "LR+Adaptive"]
    out = out.reset_index()
    out["method"] = pd.Categorical(out["method"], categories=order, ordered=True)
    return out.sort_values(["dataset", "method"]).reset_index(drop=True)


def _plot_scatter(df, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, dataset in zip(axes, ["Adult", "German"]):
        sub = df[df["dataset"] == dataset]
        for m, marker in [("LR", "o"), ("RF", "s"), ("SVM", "^"), ("LR+Adaptive", "*")]:
            d = sub[sub["method"] == m]
            ax.errorbar(d["disp_tpr"].mean(), d["acc"].mean(),
                        xerr=d["disp_tpr"].std(), yerr=d["acc"].std(),
                        fmt=marker, markersize=11, capsize=4, label=m)
        ax.set_xlabel("Disparate TPR (equal opportunity gap) — lower is better")
        ax.set_ylabel("Accuracy")
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/figures/Disparate_TPR_on_Adult_and_German.png", dpi=150)
    plt.close()


def _plot_fairness_bars(df, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    method_order = ["LR", "RF", "SVM", "LR+Adaptive"]
    gap_cols = ["disp_impact", "disp_tpr", "disp_tnr"]
    gap_labels = ["Disp. Impact", "Disp. TPR", "Disp. TNR"]
    bar_width = 0.2
    for ax, dataset in zip(axes, ["Adult", "German"]):
        sub = df[df["dataset"] == dataset]
        x = np.arange(len(gap_cols))
        for i, m in enumerate(method_order):
            d = sub[sub["method"] == m]
            means = [d[c].mean() for c in gap_cols]
            stds  = [d[c].std()  for c in gap_cols]
            ax.bar(x + i * bar_width, means, bar_width, yerr=stds,
                   label=m, capsize=3, alpha=0.85)
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(gap_labels)
        ax.set_ylabel("Gap (lower = fairer)")
        ax.set_title(dataset)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/figures/fairness_gaps.png", dpi=150)
    plt.close()
