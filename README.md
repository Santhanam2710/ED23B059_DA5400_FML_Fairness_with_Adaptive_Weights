# ED23B059 - DA5400 - Foundations of Machine Learning - Assignment - Fairness with Adaptive Weights

> A reproducible implementation of **Chai & Wang (ICML 2022) — *Fairness with Adaptive Weights*** on the **Adult Income** (UCI #2) and **German Credit** (UCI #144) classification tasks.

This repository accompanies a course-project reproduction of the paper. It provides (i) a clean re-implementation of the closed-form per-subgroup weight solver (Theorem 3.1) and the iterative training procedure (Algorithm 1), (ii) three pre-wired empirical studies — a main 4-method comparison, an α-sensitivity sweep, and a group-dependent label-noise robustness analysis — and (iii) a fully environment-variable–driven CLI so every figure and table in the report can be regenerated from a single command.

**Author.** Santa, IIT Madras (ED23B059) — Foundations of Machine Learning (DA5400) bonus assignment.

**Paper.** Junyi Chai, Xiaoqian Wang. *"Fairness with Adaptive Weights."* ICML 2022.  
[[Paper link]](https://proceedings.mlr.press/v162/chai22a.html)

**Video presentation.** `[Add your video link here]`

---

## Table of contents

1. [Highlights](#1-highlights)  
2. [Method recap](#2-method-recap)  
3. [Datasets](#3-datasets)  
4. [Repository layout](#4-repository-layout)  
5. [Installation](#5-installation)  
6. [Quickstart](#6-quickstart)  
7. [Detailed usage](#7-detailed-usage)  
8. [Outputs](#8-outputs)  
9. [Pre-computed results](#9-pre-computed-results)  
10. [Reproducibility checklist](#10-reproducibility-checklist)  
11. [Citation](#11-citation)  

---

## 1. Highlights

- **Faithful re-implementation.** Theorem 3.1 closed-form solver and Algorithm 1 iterative reweighing loop are implemented from the paper, with the same per-sample binary cross-entropy objective and the largest-subgroup-size constraint `c = max_k |G_k|`.
- **Two datasets, four methods, five seeds.** Adult Income and German Credit × {LR, RF, SVM, LR+Adaptive} × seeds {0, 1, 2, 3, 4}, all in one command.
- **Three reproducible studies.** (a) Main fairness/accuracy comparison, (b) α-sensitivity (paper Figure 4 analogue), (c) group-dependent label-flip noise robustness (paper Figure 3 analogue).
- **Single-folder output.** Every CSV, every figure, and every per-run log is written under `results/`. Nothing leaks elsewhere.
- **Env-var-driven CLI.** All parameters (seeds, α values, noise grid, experiment selector) read from `.env`; no code edits required to sweep configurations.

---

## 2. Method recap

The paper proposes a sample-reweighing scheme that, given a base classifier, alternates between two steps until convergence:

**(i) Per-subgroup weight update** — for each subgroup `G_k` indexed by `(label, sensitive)`, solve:

```
max_w   Σ_i  w_i · l_i  −  α · ‖w‖₂²
s.t.    Σ_i  w_i = c,    w_i ≥ 0
```

where `l_i` is the per-sample binary cross-entropy and `c = max_k |G_k|`. Theorem 3.1 gives a one-shot closed-form solution via a soft-threshold rule.  
→ Implemented in [`src/methods/weight_solver.py`](src/methods/weight_solver.py)

**(ii) Reweighted refit** — refit the classifier on `(X, y)` using the new sample weights via scikit-learn's `sample_weight` parameter.  
→ Implemented in [`src/methods/adaptive_reweigh.py`](src/methods/adaptive_reweigh.py)

**Evaluation metrics** (see [`src/helpers/metrics.py`](src/helpers/metrics.py)):

| Metric | Definition |
|---|---|
| Accuracy | `Pr[ŷ = y]` |
| Disparate Impact | `|Pr[ŷ=1 | s=0] − Pr[ŷ=1 | s=1]|` |
| Disparate TPR (Equal Opportunity gap) | `|TPR_{s=0} − TPR_{s=1}|` |
| Disparate TNR | `|TNR_{s=0} − TNR_{s=1}|` |

---

## 3. Datasets

Both loaders fetch data online at runtime — **no manual download required**.

| Key | Source | n | Sensitive attribute (s) | Target (y) |
|---|---|---|---|---|
| Adult Income | `ucimlrepo.fetch_ucirepo(id=2)` | ~48,842 | sex: Male=1, Female=0 | income >50K (yes=1) |
| German Credit | `ucimlrepo.fetch_ucirepo(id=144)` | 1,000 | sex from personal-status: Male=1, Female=0 | good credit (yes=1) |

**Preprocessing.** Both datasets: one-hot encode categoricals (drop first level), `StandardScaler`-normalise continuous features, drop the sensitive attribute from the feature matrix.

---

## 4. Repository layout

```
Fairness-with-Adaptive-Weights/
├── main.py                        # CLI entry-point — reads .env, dispatches experiment
├── .env.example                   # all configuration knobs (copy to .env)
├── .gitignore
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── loaders.py             # load_adult(), load_german()
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── weight_solver.py       # Theorem 3.1 closed-form solver
│   │   ├── adaptive_reweigh.py    # Algorithm 1 reweigh-refit loop
│   │   └── classifiers.py         # LR / RF / SVM sklearn factories
│   ├── helpers/
│   │   ├── __init__.py
│   │   └── metrics.py             # accuracy + 3 fairness gap metrics
│   └── experiments/
│       ├── __init__.py
│       ├── main_comparison.py     # Experiment 1: 4-method table + plots
│       ├── alpha_sweep.py         # Experiment 2: α-sensitivity
│       └── noise_robustness.py    # Experiment 3: label-noise analysis
│
├── notebooks/
│   └── ED23B059_DA5400_FML_BONUS_Assignment_Fairness_Adaptive_Weights.ipynb
│                                  # Original Colab notebook (monolithic version)
│
└── results/                       # Pre-computed outputs (reproducible via main.py)
    ├── figures/
    │   ├── fairness_gaps.png
    │   ├── Disparate_TPR_on_Adult_and_German.png
    │   ├── Alpha_Sensitivity_Plot.png
    │   ├── Change_of_accuracy_and_fairness_under_different_noise_ratio_on_Adult_and_German_datasets.png
    │   └── Experimental_Results_on_Adult_and_German.png
    └── tables/
        ├── raw_results.csv
        ├── summary_table.csv
        └── noise_robustness_results.csv
```

---

## 5. Installation

### Prerequisites

- Python 3.9+ (tested on 3.10, 3.11)
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Fairness-with-Adaptive-Weights.git
cd Fairness-with-Adaptive-Weights

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and (optionally) edit the environment config
cp .env.example .env
```

That's it — the datasets are fetched automatically on first run.

---

## 6. Quickstart

Run all three experiments with default settings:

```bash
python main.py
```

This will:
1. Download Adult and German Credit datasets from UCI (first run only — cached afterwards).
2. Train LR, RF, SVM, and LR+Adaptive across 5 seeds on both datasets.
3. Run the α-sensitivity sweep (8 α values × 5 seeds × 2 datasets).
4. Run the noise robustness analysis (7 noise levels × 4 methods × 5 seeds × 2 datasets).
5. Save all CSVs to `results/tables/` and all figures to `results/figures/`.

**Expected runtime:** ~10–15 minutes on a standard laptop (no GPU needed).

---

## 7. Detailed usage

### Running individual experiments

Edit `.env` or set environment variables:

```bash
# Run only the main comparison
EXPERIMENTS=main python main.py

# Run only the alpha sweep
EXPERIMENTS=alpha_sweep python main.py

# Run only noise robustness
EXPERIMENTS=noise_robustness python main.py

# Run a subset
EXPERIMENTS=main,alpha_sweep python main.py
```

### Changing hyperparameters

All tunable parameters live in `.env`:

```ini
# Change alpha for Adult dataset
ALPHA_ADULT=5.0

# Use 3 seeds instead of 5
SEEDS=0,1,2

# Coarser noise grid
NOISE_RATIOS=0.0,0.10,0.20,0.30
```

### Using the original Colab notebook

The monolithic notebook is preserved at `notebooks/ED23B059_DA5400_FML_BONUS_Assignment_Fairness_Adaptive_Weights.ipynb`. You can upload it directly to [Google Colab](https://colab.research.google.com/) and run all cells top-to-bottom. It is fully self-contained (installs `ucimlrepo` in the first cell).

---

## 8. Outputs

After running `python main.py`, the `results/` directory contains:

### Tables (`results/tables/`)

| File | Description |
|---|---|
| `raw_results.csv` | Per-seed metrics for all 4 methods × 2 datasets |
| `summary_table.csv` | Mean ± std aggregation (paper's reporting format) |
| `noise_robustness_results.csv` | Per-seed metrics under 7 noise levels |
| `alpha_sweep_results.csv` | Per-seed metrics across 8 α values |

### Figures (`results/figures/`)

| File | Description |
|---|---|
| `Disparate_TPR_on_Adult_and_German.png` | Accuracy vs. fairness scatter (Pareto frontier) |
| `fairness_gaps.png` | Grouped bar chart of all 3 fairness gaps |
| `Alpha_Sensitivity_Plot.png` | Accuracy and Disparate FPR vs. α |
| `Change_of_accuracy_and_fairness_under_different_noise_ratio_on_Adult_and_German_datasets.png` | 4×2 grid: all metrics under noise |
| `Experimental_Results_on_Adult_and_German.png` | Summary results table (screenshot) |

---

## 9. Pre-computed results

The `results/` directory ships with pre-computed outputs so you can inspect them without re-running. Key findings:

**Adult dataset (n ≈ 48K):**  LR+Adaptive reduces Disparate TPR from 0.088 (LR baseline) to **0.029** — a ~67% relative improvement — at the cost of ~4 percentage points of accuracy (0.812 vs 0.851).

**German Credit (n = 1000):** Improvements are smaller and noisier due to the small dataset. LR+Adaptive achieves the lowest Disparate Impact (0.062) but improvements in TPR gap are within error bars.

See the full summary:

| Dataset | Method | Accuracy | Disp. Impact | Disp. TPR | Disp. TNR |
|---------|--------|----------|--------------|-----------|-----------|
| Adult | LR | 0.8506 ± 0.0036 | 0.1738 ± 0.0085 | 0.0876 ± 0.0302 | 0.0706 ± 0.007 |
| Adult | RF | 0.8525 ± 0.0038 | 0.1791 ± 0.0123 | 0.0727 ± 0.0337 | 0.0754 ± 0.0088 |
| Adult | SVM | 0.8505 ± 0.0031 | 0.1681 ± 0.0089 | 0.0861 ± 0.0336 | 0.0657 ± 0.0064 |
| Adult | **LR+Adaptive** | 0.8121 ± 0.0062 | 0.1979 ± 0.0089 | **0.0285 ± 0.0182** | 0.0968 ± 0.0026 |
| German | LR | 0.754 ± 0.0413 | 0.0862 ± 0.0719 | 0.0719 ± 0.0267 | 0.1018 ± 0.083 |
| German | RF | 0.75 ± 0.0278 | 0.0642 ± 0.0359 | 0.0505 ± 0.0402 | 0.0369 ± 0.0268 |
| German | SVM | 0.758 ± 0.0412 | 0.0833 ± 0.0708 | 0.0647 ± 0.0191 | 0.1163 ± 0.0796 |
| German | **LR+Adaptive** | 0.707 ± 0.0432 | **0.0617 ± 0.0362** | 0.0674 ± 0.0549 | 0.0898 ± 0.0253 |

---

## 10. Reproducibility checklist

- [x] Fixed random seeds (0–4) for train/test splits
- [x] Stratified splits (80/20) preserving label distribution
- [x] Identical preprocessing pipeline across all methods
- [x] Same fairness metric definitions as the paper (Section 3.1)
- [x] Theorem 3.1 solver implemented exactly as described
- [x] Algorithm 1 convergence criterion: relative L2 change < 10⁻³ or 15 iterations
- [x] All results reported as mean ± std over 5 seeds

---

## 11. Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{chai2022fairness,
  title     = {Fairness with Adaptive Weights},
  author    = {Chai, Junyi and Wang, Xiaoqian},
  booktitle = {Proceedings of the 39th International Conference on
               Machine Learning (ICML)},
  pages     = {2853--2866},
  year      = {2022},
}
```
