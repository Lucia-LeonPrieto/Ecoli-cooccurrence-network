#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extend_tableS2_beta_boundary.py
================================
The "boundary-optimum check" cited in Methods 2.4 as Supplementary Table S2:
two extra beta_kl configs (0.005, 0.001) for the winning architecture
(latent_dim=32, hidden_dim=64), evaluated with the SAME ground-truth
composite-selection criterion (mean_auc_pr, mean_fpr_indirect,
mean_fpr_latent) and the SAME 30 tuning replicates (TUNING_SEEDS 200-229)
as the original 48-config grid in run_hyperparameter_search_ground_truth()
(simulation_ground_truth.py).

Appends to results/ground_truth_validation/TableS2.csv as config_id 49/50,
flagged in a `boundary_probe` column so the original 48-row grid stays
distinguishable from this addition. Does NOT rerun the other 46 configs --
it requires the checked-in 48-row TableS2.csv (from the one-time
`simulation_ground_truth.py --phase select-hyperparams` grid search, which
is not part of run_all.py) to already be present.

Held-out log-likelihood/ELBO for these same 2 configs (a different,
complementary, non-selecting criterion) is read from
results/ground_truth_validation/TableS4.csv, produced by
held_out_loglik_beta_kl_comparison.py -- not recomputed here.

Wired into run_all.py as step 14; must run after step 10
(held_out_loglik_beta_kl_comparison.py). Run standalone with:
    python src/extend_tableS2_beta_boundary.py
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch  # noqa: E402
torch.set_num_threads(1)

from simulation_ground_truth import (  # noqa: E402
    load_real_matrix, generate_synthetic_ground_truth, _eval_score_matrix,
    TUNING_SEEDS, SIM_KWARGS, OUT_DIR,
)
from inference_functions import bernoulli_vae_network  # noqa: E402
from vae_config_loader import load_vae_config  # noqa: E402

TABLE_S2 = Path(OUT_DIR) / "TableS2.csv"
TABLE_S4 = Path(OUT_DIR) / "TableS4.csv"

NEW_CONFIGS = [
    dict(latent_dim=32, hidden_dim=64, beta_kl=0.005),
    dict(latent_dim=32, hidden_dim=64, beta_kl=0.001),
]


def main():
    warnings.filterwarnings("ignore")
    t0 = time.time()
    cfg = load_vae_config()
    base_params = dict(cfg["base_params"])

    real = load_real_matrix()

    print("=" * 78)
    print("  BOUNDARY-OPTIMUM CHECK -- extending Table S2's 48-config grid with")
    print(f"  beta_kl in {[c['beta_kl'] for c in NEW_CONFIGS]} (arch 32/64), "
          f"tuning seeds {TUNING_SEEDS[0]}-{TUNING_SEEDS[-1]} ({len(TUNING_SEEDS)} replicas)")
    print("=" * 78)

    tuning_data = [
        (seed,) + generate_synthetic_ground_truth(seed=seed, real_matrix=real, **SIM_KWARGS)
        for seed in TUNING_SEEDS
    ]

    if not TABLE_S2.is_file():
        raise SystemExit(
            f"{TABLE_S2} not found -- this script appends to the checked-in "
            "48-config grid produced by `simulation_ground_truth.py --phase "
            "select-hyperparams` (a one-time step not run by run_all.py); "
            "restore that file before rerunning."
        )
    existing = pd.read_csv(TABLE_S2)
    if "boundary_probe" in existing.columns:
        # idempotent: drop any boundary-probe rows from a prior run of this
        # script before recomputing, so rerunning doesn't duplicate them
        existing = existing[~existing["boundary_probe"].astype(bool)].copy()
    next_config_id = int(existing["config_id"].max()) + 1

    new_rows = []
    for hp in NEW_CONFIGS:
        vae_params = {**base_params, **hp}
        print(f"\n  config {next_config_id:02d} (boundary probe): {hp}")
        aucs, fpr_is, fpr_ls = [], [], []
        for seed, bin_df, gt in tuning_data:
            n = bin_df.shape[0]
            try:
                adj_vae, _, _ = bernoulli_vae_network(
                    bin_df, normalize_adj=False, verbose=False, seed=0, **vae_params
                )
                S = np.asarray(adj_vae.values, dtype=float)
                auc, fpr_i, fpr_l, failed = _eval_score_matrix(S, gt, n)
            except Exception:
                auc, fpr_i, fpr_l, failed = np.nan, np.nan, np.nan, True
            if not failed:
                aucs.append(auc)
                fpr_is.append(fpr_i)
                fpr_ls.append(fpr_l)
        row = dict(
            config_id=next_config_id, **hp,
            n_replicas_ok=len(aucs),
            mean_auc_pr=float(np.mean(aucs)) if aucs else np.nan,
            std_auc_pr=float(np.std(aucs)) if aucs else np.nan,
            mean_fpr_indirect=float(np.mean(fpr_is)) if fpr_is else np.nan,
            std_fpr_indirect=float(np.std(fpr_is)) if fpr_is else np.nan,
            mean_fpr_latent=float(np.mean(fpr_ls)) if fpr_ls else np.nan,
            std_fpr_latent=float(np.std(fpr_ls)) if fpr_ls else np.nan,
        )
        print(f"    mean_auc_pr={row['mean_auc_pr']:.4f}  mean_fpr_latent={row['mean_fpr_latent']:.4f}  "
              f"n_replicas_ok={row['n_replicas_ok']}/{len(TUNING_SEEDS)}")
        new_rows.append(row)
        next_config_id += 1

    new_df = pd.DataFrame(new_rows)

    # merge held-out log-likelihood/ELBO for the same 2 configs from TableS4.csv
    # (same seed=42, k_folds=5, already computed -- not recomputed here)
    s4 = pd.read_csv(TABLE_S4)
    merge_keys = ["latent_dim", "hidden_dim", "beta_kl"]
    new_df = new_df.merge(
        s4[merge_keys + ["mean_elbo_held_out", "mean_log_likelihood_held_out"]],
        on=merge_keys, how="left",
    )

    existing["boundary_probe"] = False
    new_df["boundary_probe"] = True

    # composite score, computed the SAME way as _select_winner_and_write_config,
    # but min-max normalized across the ORIGINAL 48-config set only (not
    # including these 2 boundary probes): the probe's score is evaluated
    # AGAINST the already-fixed 48-config normalization range, not
    # re-normalized including itself.
    lo_auc, hi_auc = existing["mean_auc_pr"].min(), existing["mean_auc_pr"].max()
    lo_spec, hi_spec = (1 - existing["mean_fpr_latent"]).min(), (1 - existing["mean_fpr_latent"]).max()

    def composite(row):
        norm_auc = (row["mean_auc_pr"] - lo_auc) / (hi_auc - lo_auc)
        spec = 1 - row["mean_fpr_latent"]
        norm_spec = (spec - lo_spec) / (hi_spec - lo_spec)
        if norm_auc + norm_spec == 0:
            return 0.0
        return 2 * norm_auc * norm_spec / (norm_auc + norm_spec)

    existing["score_vs_48grid_norm"] = existing.apply(composite, axis=1)
    new_df["score_vs_48grid_norm"] = new_df.apply(composite, axis=1)

    combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    combined.to_csv(TABLE_S2, index=False)

    winner_01 = existing.loc[existing["beta_kl"] == 0.01].iloc[0]
    print("\n" + "=" * 78)
    print("  RESULT")
    print("=" * 78)
    for _, row in new_df.iterrows():
        print(f"  beta_kl={row['beta_kl']}: mean_auc_pr={row['mean_auc_pr']:.4f}  "
              f"mean_fpr_latent={row['mean_fpr_latent']:.4f}  "
              f"score(vs 48-grid norm)={row['score_vs_48grid_norm']:.4f}")
    print(f"  beta_kl=0.01 (current winner): mean_auc_pr={winner_01['mean_auc_pr']:.4f}  "
          f"mean_fpr_latent={winner_01['mean_fpr_latent']:.4f}  "
          f"score(vs 48-grid norm)={winner_01['score_vs_48grid_norm']:.4f}")
    print(f"\n  -> {TABLE_S2} updated in place (48 original rows + 2 boundary-probe rows, "
          f"flagged in 'boundary_probe' column)")
    print(f"\n  total runtime: {(time.time() - t0) / 60:.2f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
