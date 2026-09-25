#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
threshold_sensitivity_ground_truth.py
=======================================
Threshold-sensitivity analysis for the reported network's top-30% edge cut,
using the SAME simulated-ground-truth machinery as Table 1
(simulation_ground_truth.py: generate_synthetic_ground_truth, 35 replicates,
seeds 100-134 -- the FINAL EVALUATION block, same one used to report Table 1
numbers) and the FINAL operational VAE hyperparameters
(src/vae_config.json, latent_dim=32, hidden_dim=64, beta_kl=0.01).

This is DIFFERENT from the pre-existing results/02_threshold_sensitivity.csv /
01_threshold_sensitivity_bootstrap_mean.csv, which sweep the percentile cut on
the REAL network and report edge count / isolated nodes / min degree --
those say nothing about ACCURACY, because the real data has no known ground
truth to check recovery against. This script instead asks: at each
candidate top-k% cut, how well would the VAE's recovered edge set have
performed against the KNOWN direct/indirect/latent structure of the
simulation? That is the evidence needed to argue the 30% choice is not
arbitrary.

For percentiles in [20, 25, 30, 35, 40] (i.e. keeping the top 20-40% of
pairs by |score|), and for each of the 35 replicates already used for
Table 1:
  - auc_pr is threshold-INDEPENDENT (rank-based over ALL scored pairs), so
    it is identical across percentiles for a fixed replicate -- included at
    every row anyway for completeness / so a reader doesn't have to look
    elsewhere to confirm it doesn't change.
  - fpr_indirect / fpr_latent DO depend on the percentile cut (they are
    "fraction of that category found in the top-k%"), recomputed here via
    the same _fpr_in_category() logic already used in
    simulation_ground_truth.py, generalized to a variable top_frac instead
    of the hardcoded 0.30.

Inputs:
    data/input/EffectorRepertoire_Ecoli.xlsx
    src/vae_config.json
Output:
    results/ground_truth_validation/TableS6.csv
        one row per (percentile, seed): auc_pr, fpr_indirect, fpr_latent
        (this is the manuscript's Table S6)
    results/ground_truth_validation/11_threshold_sensitivity_ground_truth_summary.csv
        one row per percentile: mean/std auc_pr, fpr_indirect, fpr_latent
        across the 35 replicates

Wired into run_all.py as step 11 (Table S6, cited in Methods 2.7), so a
clean `run_all.py` reproduces this number without a separate manual run.
Fully self-contained: regenerates its own synthetic ground truth from the
same 35 replicates as Table 1, with no dependency on any other step's
output files. Run standalone with:
    python src/threshold_sensitivity_ground_truth.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import simulation_ground_truth as sgt  # noqa: E402
from inference_functions import bernoulli_vae_network  # noqa: E402
from vae_config_loader import load_vae_params  # noqa: E402

OUT_LONG = ROOT / "results" / "ground_truth_validation" / "TableS6.csv"
OUT_SUMMARY = ROOT / "results" / "ground_truth_validation" / "11_threshold_sensitivity_ground_truth_summary.csv"

PERCENTILES = [20, 25, 30, 35, 40]  # top-k% kept, i.e. top_frac = pct / 100


def _eval_at_percentile(S, gt, n, top_frac):
    """Same protocol as simulation_ground_truth.py's _eval_score_matrix(),
    generalized to an arbitrary top_frac instead of the hardcoded 0.30."""
    all_pairs = sgt._upper_pairs(n)
    pair_index = {pr: k for k, pr in enumerate(all_pairs)}
    pos = {(i, j) for i, j, _ in gt["direct"]}
    indir = [(i, j) for i, j in gt["indirect"]]
    lat = [(i, j) for i, j in gt["latent"]]
    neg = set(indir) | set(lat) | set(gt["noise"])
    y_true = np.array([1 if pr in pos else (0 if pr in neg else -1) for pr in all_pairs])
    keep = y_true >= 0
    y_eval = y_true[keep]
    pairs_eval = [pr for pr, k in zip(all_pairs, keep) if k]

    svec_all = np.array([abs(S[i, j]) for i, j in all_pairs])
    svec_eval = np.array([abs(S[i, j]) for i, j in pairs_eval])
    auc = sgt._auc_pr(y_eval, svec_eval)
    fpr_i = sgt._fpr_in_category(svec_all, pair_index, indir, all_pairs, top_frac)
    fpr_l = sgt._fpr_in_category(svec_all, pair_index, lat, all_pairs, top_frac)
    return auc, fpr_i, fpr_l


def main() -> int:
    t0 = time.time()
    warnings.filterwarnings("ignore")

    real = sgt.load_real_matrix()
    vae_params = load_vae_params()
    print(f"vae_params: {vae_params}")
    print(f"replicates: seeds {sgt.PHASE2_SEEDS[0]}..{sgt.PHASE2_SEEDS[-1]} "
          f"({len(sgt.PHASE2_SEEDS)}, same as Table 1)")
    print(f"percentiles: {PERCENTILES}")

    rows = []
    for rep, seed in enumerate(sgt.PHASE2_SEEDS, 1):
        bin_df, gt = sgt.generate_synthetic_ground_truth(seed=seed, real_matrix=real, **sgt.SIM_KWARGS)
        n = bin_df.shape[0]
        adj_vae, _, _ = bernoulli_vae_network(
            bin_df, normalize_adj=False, verbose=False, seed=0, **vae_params
        )
        S = np.asarray(adj_vae.values, dtype=float)

        for pct in PERCENTILES:
            auc, fpr_i, fpr_l = _eval_at_percentile(S, gt, n, top_frac=pct / 100.0)
            rows.append(dict(percentile=pct, seed=seed, auc_pr=auc,
                             fpr_indirect=fpr_i, fpr_latent=fpr_l))
        print(f"  replicate {rep:>2}/{len(sgt.PHASE2_SEEDS)} (seed {seed}) done")

    long_df = pd.DataFrame(rows)
    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_LONG, index=False)

    summary_df = long_df.groupby("percentile").agg(
        mean_auc_pr=("auc_pr", "mean"), std_auc_pr=("auc_pr", "std"),
        mean_fpr_indirect=("fpr_indirect", "mean"), std_fpr_indirect=("fpr_indirect", "std"),
        mean_fpr_latent=("fpr_latent", "mean"), std_fpr_latent=("fpr_latent", "std"),
        n_replicates=("seed", "count"),
    ).reset_index()
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("\n" + "=" * 78)
    print("  SUMMARY -- threshold sensitivity on simulated ground truth")
    print("=" * 78)
    print(summary_df.to_string(index=False))

    print(f"\n  -> {OUT_LONG.relative_to(ROOT)}")
    print(f"  -> {OUT_SUMMARY.relative_to(ROOT)}")
    print(f"\n  done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
