#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_lineage_control.py
==========================
Supplementary control for phylogenetic_sensitivity.py's leave-one-lineage-out
analysis: is the real lineage exclusion's jaccard_red_final_top30 actually
different from excluding a RANDOM group of strains of the same size, or is
it indistinguishable from what any random subset removal would do?

For the two largest lineage clusters (75 and 52 strains -- the ones with the
lowest jaccard_red_final_top30 in 06_leave_one_lineage_out.csv, 0.3205 and
0.4498 respectively), draws 15 random same-size strain subsets each (30
single-shot VAE fits total, ~12s/fit on this machine -- a few minutes, no
bootstrap, no retraining of anything else in the pipeline), excludes each,
rebuilds the network with the SAME single-shot method and the SAME
count-based top-30% threshold recomputed on the reduced data
(own_top30_edge_set, borrowed from phylogenetic_sensitivity.py's logic), and
compares each to the reported network (02_network_final_top30.json) via
Jaccard.

Seeds: RNG_SEED_BASE + size + replicate index (e.g. 10075+0..14 for size 75,
10052+0..14 for size 52) -- distinct, reproducible, and independent of the
main pipeline's SEED=42 (which is reserved for the single-shot VAE fit
itself, kept fixed across all 30 replicates so the ONLY thing varying is
which strains are excluded, not the VAE's own stochasticity).

Inputs (relative to the repository root):
    data/input/EffectorRepertoire_Ecoli.xlsx
    results/02_network_final_top30.json
    src/vae_config.json
Output:
    results/phylogenetic_sensitivity/06b_random_lineage_control.csv

Usage:
    python src/random_lineage_control.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from inference_functions import load_binary_matrix, bernoulli_vae_network  # noqa: E402
from vae_config_loader import load_vae_params  # noqa: E402

INPUT_XLSX = ROOT / "data" / "input" / "EffectorRepertoire_Ecoli.xlsx"
REF_JSON = ROOT / "results" / "02_network_final_top30.json"
OUT_CSV = ROOT / "results" / "phylogenetic_sensitivity" / "06b_random_lineage_control.csv"

VAE_PARAMS = load_vae_params()
SEED = 42          # same as build_final_network.py / phylogenetic_sensitivity.py
N_REPLICATES = 15
SIZES = [75, 52]   # the two largest real lineage clusters
RNG_SEED_BASE = 10000

REAL_LINEAGE_JACCARD = {75: 0.3205, 52: 0.4498}  # from 06_leave_one_lineage_out.csv, for reference


def own_top30_edge_set(arr, effectors, top_percent=30):
    """Identical procedure to phylogenetic_sensitivity.py's helper of the same
    purpose: count-based top-30% cut on |weight|, excluding exact zeros,
    threshold = k-th largest, retain >= threshold."""
    ii, jj = np.triu_indices(arr.shape[0], k=1)
    abs_tri = np.abs(arr[ii, jj])
    nonzero = abs_tri[abs_tri > 0]
    k = max(int(len(nonzero) * top_percent / 100), 1)
    threshold = np.sort(nonzero)[-k]
    return {
        tuple(sorted((effectors[ii[t]], effectors[jj[t]])))
        for t in np.where(abs_tri >= threshold)[0]
    }


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else np.nan


def main() -> int:
    t0 = time.time()
    matrix = load_binary_matrix(INPUT_XLSX)
    effectors = matrix.index.tolist()
    strains = matrix.columns.tolist()
    n_total = len(strains)

    with open(REF_JSON, encoding="utf-8") as fh:
        ref_net = json.load(fh)
    json_pairs = {
        tuple(sorted((e["data"]["source"], e["data"]["target"])))
        for e in ref_net["elements"]["edges"]
    }
    print(f"reference network: {len(json_pairs)} edges ({REF_JSON.relative_to(ROOT)})")
    print(f"vae_params: {VAE_PARAMS}")

    rows = []
    for size in SIZES:
        print(f"\n=== size={size} ({N_REPLICATES} random exclusions) ===")
        for rep in range(N_REPLICATES):
            rng_seed = RNG_SEED_BASE + size + rep
            rng = np.random.default_rng(rng_seed)
            excl_idx = rng.choice(n_total, size=size, replace=False)
            excl_strains = [strains[i] for i in excl_idx]
            sub = matrix.drop(columns=excl_strains)

            t1 = time.time()
            adj_df, _, _ = bernoulli_vae_network(
                sub, **VAE_PARAMS, device="cpu", seed=SEED, verbose=False,
                normalize_adj=False,
            )
            own_top30 = own_top30_edge_set(adj_df.values, effectors, top_percent=30)
            jac = float(jaccard(json_pairs, own_top30))

            rows.append(dict(
                size=size, replicate=rep, rng_seed=rng_seed,
                n_strains_remaining=sub.shape[1],
                n_edges_own_top30=len(own_top30),
                jaccard_red_final_top30=round(jac, 4),
            ))
            print(f"  rep {rep + 1:>2}/{N_REPLICATES} (seed {rng_seed}): "
                  f"jaccard={jac:.4f}  ({time.time() - t1:.1f}s)")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 70)
    print("  SUMMARY -- random-exclusion control vs. real lineage exclusion")
    print("=" * 70)
    for size in SIZES:
        sub = df[df["size"] == size]["jaccard_red_final_top30"]
        real = REAL_LINEAGE_JACCARD[size]
        mean_r, std_r = sub.mean(), sub.std()
        lo, hi = sub.min(), sub.max()
        z = (real - mean_r) / std_r if std_r > 0 else np.nan
        print(f"\n  size={size}:")
        print(f"    random exclusions (n={len(sub)}): mean={mean_r:.4f}  "
              f"std={std_r:.4f}  range=[{lo:.4f}, {hi:.4f}]")
        print(f"    real lineage exclusion: {real:.4f}")
        if real < lo:
            verdict = f"BELOW the random range (real < min random by {lo - real:.4f})"
        elif real > hi:
            verdict = f"ABOVE the random range (real > max random by {real - hi:.4f})"
        else:
            verdict = "WITHIN the random range"
        print(f"    verdict: {verdict}  (z-score vs. random distribution: {z:.2f})")

    print(f"\n  -> {OUT_CSV.relative_to(ROOT)}")
    print(f"  done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
