#!/usr/bin/env python3
"""CLI entry-point for Fairness with Adaptive Weights experiments.

Reads configuration from ``.env`` (or environment variables) and
dispatches the requested experiment(s).

Usage
-----
    cp .env.example .env   # edit if needed
    python main.py
"""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


def _parse_list(key, default, cast=str):
    raw = os.getenv(key, default)
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    # ── Parse configuration ──────────────────────────────────────
    experiments  = _parse_list("EXPERIMENTS", "all")
    seeds        = _parse_list("SEEDS", "0,1,2,3,4", int)
    test_size    = float(os.getenv("TEST_SIZE", "0.2"))
    alpha_adult  = float(os.getenv("ALPHA_ADULT", "10.0"))
    alpha_german = float(os.getenv("ALPHA_GERMAN", "2.0"))
    alpha_grid   = _parse_list("ALPHA_GRID",
                               "0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0", float)
    noise_ratios = _parse_list("NOISE_RATIOS",
                               "0.0,0.05,0.10,0.15,0.20,0.25,0.30", float)
    results_dir  = os.getenv("RESULTS_DIR", "results")

    run_all = "all" in experiments

    # ── Ensure output directories exist ──────────────────────────
    Path(f"{results_dir}/figures").mkdir(parents=True, exist_ok=True)
    Path(f"{results_dir}/tables").mkdir(parents=True, exist_ok=True)

    # ── Dispatch ─────────────────────────────────────────────────
    if run_all or "main" in experiments:
        print("=" * 60)
        print("EXPERIMENT 1 — Main 4-method comparison")
        print("=" * 60)
        from src.experiments.main_comparison import run as run_main
        run_main(seeds, test_size, alpha_adult, alpha_german, results_dir)

    if run_all or "alpha_sweep" in experiments:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2 — Alpha sensitivity sweep")
        print("=" * 60)
        from src.experiments.alpha_sweep import run as run_alpha
        run_alpha(seeds, test_size, alpha_grid, results_dir)

    if run_all or "noise_robustness" in experiments:
        print("\n" + "=" * 60)
        print("EXPERIMENT 3 — Noise robustness")
        print("=" * 60)
        from src.experiments.noise_robustness import run as run_noise
        run_noise(seeds, test_size, noise_ratios, results_dir)

    print("\n✅  All requested experiments complete.")
    print(f"    Results saved to: {results_dir}/")


if __name__ == "__main__":
    main()
