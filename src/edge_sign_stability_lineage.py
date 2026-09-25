#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_sign_stability_lineage.py
================================
Supplementary robustness check: for every one of the 585 edges in the
reported network (02_network_final_top30.json), does its SIGN (positive /
negative co-occurrence) survive leave-one-lineage-out exclusion?

Reuses the 5 cached lineage-excluded single-shot VAE reconstructions already
computed by phylogenetic_sensitivity.py (results/phylogenetic_sensitivity/
06_consensus_excl_cluster{1..5}.npy) -- no retraining.

Inputs (relative to the repository root):
    results/02_network_final_top30.json                          -> the 585 edges
    results/phylogenetic_sensitivity/06_consensus_excl_cluster{1..5}.npy
    results/TableS9.csv                                        -> k=7 module labels
Output:
    results/phylogenetic_sensitivity/08_edge_sign_stability_lineage.csv

Usage:
    python src/edge_sign_stability_lineage.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from inference_functions import load_binary_matrix  # noqa: E402

REF_JSON = ROOT / "results" / "02_network_final_top30.json"
PHYLO_DIR = ROOT / "results" / "phylogenetic_sensitivity"
CLUSTER_CSV = ROOT / "results" / "TableS9.csv"
OUT_CSV = PHYLO_DIR / "08_edge_sign_stability_lineage.csv"

N_CLUSTERS = 5


def main() -> int:
    with open(REF_JSON, encoding="utf-8") as fh:
        net = json.load(fh)
    nodes = [n["data"]["id"] for n in net["elements"]["nodes"]]
    idx = {name: i for i, name in enumerate(nodes)}
    edges = net["elements"]["edges"]
    print(f"reference network: {len(nodes)} nodes, {len(edges)} edges")

    excl_arrays = {}
    for c in range(1, N_CLUSTERS + 1):
        p = PHYLO_DIR / f"06_consensus_excl_cluster{c}.npy"
        excl_arrays[c] = np.load(p)
        print(f"  loaded 06_consensus_excl_cluster{c}.npy")

    clu_df = pd.read_csv(CLUSTER_CSV)
    module_of = dict(zip(clu_df["effector"], clu_df["cluster"]))

    def sign(x):
        return "pos" if x >= 0 else "neg"

    rows = []
    for e in edges:
        a, b = e["data"]["source"], e["data"]["target"]
        i, j = idx[a], idx[b]
        w_ref = float(e["data"]["weight"])
        s_ref = sign(w_ref)

        excl_signs = {}
        excl_weights = {}
        for c in range(1, N_CLUSTERS + 1):
            w_c = float(excl_arrays[c][i, j])
            excl_signs[c] = sign(w_c)
            excl_weights[c] = w_c

        all_signs = [s_ref] + [excl_signs[c] for c in range(1, N_CLUSTERS + 1)]
        consistent = len(set(all_signs)) == 1
        n_flips = sum(1 for c in range(1, N_CLUSTERS + 1) if excl_signs[c] != s_ref)

        row = {
            "eff_a": a, "eff_b": b,
            "weight_ref": round(w_ref, 4), "sign_ref": s_ref,
            "module_a": module_of.get(a), "module_b": module_of.get(b),
        }
        for c in range(1, N_CLUSTERS + 1):
            row[f"sign_excl_cluster{c}"] = excl_signs[c]
            row[f"weight_excl_cluster{c}"] = round(excl_weights[c], 4)
        row["sign_consistent_all5"] = consistent
        row["n_sign_flips_of_5"] = n_flips
        rows.append(row)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    n_total = len(df)
    n_consistent = int(df["sign_consistent_all5"].sum())
    print("\n" + "=" * 70)
    print("  SUMMARY -- sign stability under leave-one-lineage-out (585 edges)")
    print("=" * 70)
    print(f"  edges with consistent sign in all 5 exclusions : {n_consistent}/{n_total} "
          f"({100 * n_consistent / n_total:.2f}%)")
    print(f"  edges with >=1 sign flip                        : {n_total - n_consistent}/{n_total} "
          f"({100 * (n_total - n_consistent) / n_total:.2f}%)")

    flips = df[~df["sign_consistent_all5"]]
    if len(flips):
        print("\n  distribution of n_sign_flips_of_5 among unstable edges:")
        print(flips["n_sign_flips_of_5"].value_counts().sort_index().to_string())

        print("\n  module-pair breakdown of sign-flipping edges (module_a, module_b):")
        pair_counts = flips.groupby(["module_a", "module_b"]).size().sort_values(ascending=False)
        print(pair_counts.to_string())

        print("\n  module-pair breakdown of ALL 585 edges (for base-rate comparison):")
        all_pair_counts = df.groupby(["module_a", "module_b"]).size().sort_values(ascending=False)
        print(all_pair_counts.to_string())

    print(f"\n  -> {OUT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
