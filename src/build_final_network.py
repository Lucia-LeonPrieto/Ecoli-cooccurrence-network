#!/usr/bin/env python
"""
build_final_network.py
======================
Builds the **final reported co-occurrence network** of the manuscript and its
per-edge confidence, from the 63 x 150 binary effector matrix.

Method (fixed, not re-optimised):

  * A single Bernoulli-VAE is fitted on the **full 150-strain** matrix with the
    hyper-parameters selected by ground-truth validation and stored in
    src/vae_config.json (see vae_config_loader.load_vae_params(); 300 epochs,
    geometric must-link schedule off).  The signed raw cosine similarity
    between decoder
    weight rows is the network  ->  results/02_network_final_signed_similarity.csv
    ("single-shot VAE" network).

  * The **same** VAE is refitted on 75 bootstrap resamples of the strains
    (sample_fraction=0.6, replace=True, seed=42).  The per-edge standard
    deviation across those 75 replicas is the confidence estimate.  The stack
    is NOT averaged into the reported network; it only yields sd_bootstrap
       ->  results/02_edge_confidence.csv
       ->  results/02_S_boot_reference_stack.npy   (75, 63, 63)

  * The reported network is thresholded at the top 30 % of non-zero edges by
    |weight| (count-based cut, inclusive)
       ->  results/02_network_final_top30.json      (Cytoscape elements format)
       ->  results/02_threshold_sensitivity.csv     (percentile sweep 60..80)

All paths are relative to the repository root; nothing outside the repo is read.

Usage:
    python src/build_final_network.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

try:  # Windows console -> force UTF-8 for the "->" characters printed by helpers
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("MPLBACKEND", "Agg")
# Deterministic and, on tiny models like this VAE, much faster than the default
# many-thread config (which oversubscribes and thrashes on a busy workstation).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch  # noqa: E402
torch.set_num_threads(1)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from inference_functions import (  # noqa: E402
    load_binary_matrix,
    bootstrap_vae_network,
    export_grafo_json_topk,
)
from vae_config_loader import load_vae_params  # noqa: E402

# ── configuration ────────────────────────────────────────────────────────────
INPUT_XLSX = os.path.join(ROOT, "data", "input", "EffectorRepertoire_Ecoli.xlsx")
RESULTS_DIR = os.path.join(ROOT, "results")

SHEET_NAME = "Ecoli"
TOTAL_STRAINS = 150
SEED = 42
N_BOOTSTRAP = 75
SAMPLE_FRACTION = 0.6
TOP_PERCENT = 30
THRESHOLD_PERCENTILES = [60, 65, 68, 69, 70, 71, 72, 75, 80]

# VAE hyperparameters come from the single shared file src/vae_config.json
# (selected by ground-truth validation in simulation_ground_truth.py),
# loaded via vae_config_loader.load_vae_params() -- not hardcoded here and
# not re-optimised here, so every script that fits a VAE uses the same
# configuration.
VAE_PARAMS = load_vae_params()
# ─────────────────────────────────────────────────────────────────────────────


def _threshold_sensitivity(abs_matrix: np.ndarray) -> pd.DataFrame:
    """Percentile sweep, identical criterion to the JSON top-k export."""
    upper_idx = np.triu_indices_from(abs_matrix, k=1)
    upper_abs = abs_matrix[upper_idx]
    nonzero_sorted = np.sort(upper_abs[upper_abs > 0])

    rows = []
    for pct in THRESHOLD_PERCENTILES:
        k_edges = max(int(len(nonzero_sorted) * (100 - pct) / 100), 1)
        threshold = float(nonzero_sorted[-k_edges])
        adjacency = ((abs_matrix >= threshold) & (abs_matrix > 0)).astype(int)
        np.fill_diagonal(adjacency, 0)
        degree = adjacency.sum(axis=1)
        rows.append({
            "percentile": pct,
            "threshold": round(threshold, 6),
            "n_edges": int(adjacency[upper_idx].sum()),
            "n_isolated": int(np.sum(degree == 0)),
            "min_degree": int(degree.min()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 64)
    print("  BUILD FINAL NETWORK  (single-shot VAE + 75x bootstrap confidence)")
    print("=" * 64)

    matrix = load_binary_matrix(INPUT_XLSX, sheet_name=SHEET_NAME)
    effectors = matrix.index.tolist()
    n = len(effectors)
    effector_count = matrix.sum(axis=1)
    print(f"  matrix            : {matrix.shape[0]} effectors x {matrix.shape[1]} strains")
    print(f"  vae_params        : {VAE_PARAMS}")
    print(f"  bootstrap         : B={N_BOOTSTRAP}, sample_fraction={SAMPLE_FRACTION}, seed={SEED}")

    # One call returns: consensus (bootstrap mean, unused here),
    # S_full_df (single-shot VAE on the 150 strains = the reported network),
    # S_boot (75, n, n) raw replica stack, matrices_individuales (unused).
    _consensus, S_full_df, S_boot, _mats = bootstrap_vae_network(
        matrix,
        n_bootstrap=N_BOOTSTRAP,
        sample_fraction=SAMPLE_FRACTION,
        vae_params=VAE_PARAMS,
        normalize_for_stats=False,
        device="cpu",
        seed=SEED,
        verbose=True,
    )
    S_boot = np.asarray(S_boot, dtype=float)
    S_full_df = S_full_df.loc[effectors, effectors]

    # 1) reported network (63 x 63 signed raw cosine) ------------------------
    net_path = os.path.join(RESULTS_DIR, "02_network_final_signed_similarity.csv")
    S_full_df.to_csv(net_path)

    # 2) 75-replica reference stack ----------------------------------------
    stack_path = os.path.join(RESULTS_DIR, "02_S_boot_reference_stack.npy")
    np.save(stack_path, S_boot)

    # 3) per-edge confidence (1953 upper-triangle pairs) ------------------
    tri_i, tri_j = np.triu_indices(n, k=1)
    single_shot = S_full_df.values[tri_i, tri_j]
    sd_bootstrap = S_boot[:, tri_i, tri_j].std(axis=0)  # ddof=0, matches library
    conf_df = pd.DataFrame({
        "efector_A": [effectors[i] for i in tri_i],
        "efector_B": [effectors[j] for j in tri_j],
        "valor_disparo_unico": single_shot,
        "sd_bootstrap": sd_bootstrap,
    })
    conf_path = os.path.join(RESULTS_DIR, "02_edge_confidence.csv")
    conf_df.to_csv(conf_path, index=False)

    # 4) threshold sensitivity sweep -------------------------------------
    abs_full = np.abs(S_full_df.values)
    ts_df = _threshold_sensitivity(abs_full)
    ts_path = os.path.join(RESULTS_DIR, "02_threshold_sensitivity.csv")
    ts_df.to_csv(ts_path, index=False)

    # 5) thresholded network, Cytoscape elements JSON -------------------
    #    export_grafo_json_topk writes effectornetwork_top{p}_{filename}.json;
    #    rename to the canonical repo name afterwards.
    export_grafo_json_topk(
        S_full_df,
        effector_count,
        TOTAL_STRAINS,
        RESULTS_DIR,
        top_percent=TOP_PERCENT,
        filename="final",
        use_absolute_weights=True,
    )
    tmp_json = os.path.join(RESULTS_DIR, f"effectornetwork_top{TOP_PERCENT}_final.json")
    json_path = os.path.join(RESULTS_DIR, "02_network_final_top30.json")
    os.replace(tmp_json, json_path)

    with open(json_path, encoding="utf-8") as fh:
        net = json.load(fh)
    edges = net["elements"]["edges"]
    n_pos = sum(1 for e in edges if e["data"]["weight"] >= 0)
    n_neg = sum(1 for e in edges if e["data"]["weight"] < 0)
    min_abs = min(abs(e["data"]["weight"]) for e in edges)

    print("\n" + "-" * 64)
    print("  OUTPUTS")
    print("-" * 64)
    for p in (net_path, conf_path, stack_path, ts_path, json_path):
        print(f"  {os.path.relpath(p, ROOT)}")
    print("\n  reported network (top 30%):")
    print(f"    nodes            : {n}")
    print(f"    edges            : {len(edges)}")
    print(f"    positive / negative : {n_pos} / {n_neg}")
    print(f"    |weight| threshold  : {min_abs:.6f}")
    print(f"\n  done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
