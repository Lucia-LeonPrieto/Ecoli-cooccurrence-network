#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
effector_lineage_distribution.py
==================================
Reviewer-1 robustness check: is an effector's involvement in a co-occurrence
/ exclusion edge (e.g. EspM-EspM1, NleG-NleD) plausibly explained by the
effector simply being confined to one phylogenetic lineage / prophage
background, rather than a real functional relationship? Before interpreting
any such edge, this reports how each of the 63 effectors' carrier strains
are distributed across the 5 lineage groups already defined for the
leave-one-lineage-out analysis.

Reuses the EXACT SAME strain -> lineage assignment as
phylogenetic_sensitivity.py: reads its output artifact
(results/phylogenetic_sensitivity/06_lineage_clusters.csv) directly rather than
recomputing the genomic-distance clustering independently, so the two
analyses can never silently disagree about which strain belongs to which
lineage. This script does NOT retrain or refit anything -- it is a direct
tally over the existing binary presence/absence matrix (Block 1 logic,
load_binary_matrix) crossed with that existing lineage assignment.

Inputs (relative to the repository root):
    data/input/EffectorRepertoire_Ecoli.xlsx           -> binary matrix (Block 1 logic)
    results/phylogenetic_sensitivity/06_lineage_clusters.csv  -> strain -> lineage_cluster
                                                              (produced by phylogenetic_sensitivity.py
                                                              phase1(); this script requires it to
                                                              already exist)
Outputs:
    results/07_effector_lineage_distribution.csv          -> long format, one row per
                                                            (effector, lineage) pair
    results/07_effector_lineage_distribution_summary.csv   -> one row per effector (63 rows),
                                                            n_lineages_with_carrier,
                                                            max_lineage_share, total_carriers

Usage:
    python src/effector_lineage_distribution.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from inference_functions import load_binary_matrix  # noqa: E402

INPUT_XLSX = ROOT / "data" / "input" / "EffectorRepertoire_Ecoli.xlsx"
LINEAGE_CSV = ROOT / "results" / "phylogenetic_sensitivity" / "06_lineage_clusters.csv"
OUT_LONG = ROOT / "results" / "07_effector_lineage_distribution.csv"
OUT_SUMMARY = ROOT / "results" / "07_effector_lineage_distribution_summary.csv"


def main() -> int:
    t0 = time.time()

    if not LINEAGE_CSV.is_file():
        raise FileNotFoundError(
            f"{LINEAGE_CSV} not found -- run phylogenetic_sensitivity.py first "
            "(this script reuses its strain -> lineage assignment and does not "
            "recompute it independently)."
        )

    matrix = load_binary_matrix(INPUT_XLSX)  # effectors x strains, 0/1
    effectors = matrix.index.tolist()
    print(f"matrix: {matrix.shape[0]} effectors x {matrix.shape[1]} strains")

    lineage_df = pd.read_csv(LINEAGE_CSV)
    strain_to_lineage = dict(zip(lineage_df["strain"], lineage_df["lineage_cluster"]))
    lineages = sorted(lineage_df["lineage_cluster"].unique())
    lineage_sizes = lineage_df["lineage_cluster"].value_counts().sort_index()
    print(f"lineages (from {LINEAGE_CSV.relative_to(ROOT)}): "
          + ", ".join(f"C{c}(n={lineage_sizes[c]})" for c in lineages))

    # map matrix columns (strain ids) -> lineage; matrix columns must be a
    # subset of the strains in 06_lineage_clusters.csv (same 150-strain cohort)
    missing = [s for s in matrix.columns if s not in strain_to_lineage]
    if missing:
        raise RuntimeError(
            f"{len(missing)} strain(s) in the binary matrix have no lineage "
            f"assignment in {LINEAGE_CSV.name}: {missing[:5]}..."
        )
    strain_lineage = pd.Series({s: strain_to_lineage[s] for s in matrix.columns})

    long_rows = []
    summary_rows = []
    for eff in effectors:
        carriers = matrix.columns[matrix.loc[eff] == 1]
        total_carriers = len(carriers)
        carrier_lineage_counts = strain_lineage.loc[carriers].value_counts()

        per_lineage_shares = []
        for c in lineages:
            n_strains_lineage = int(lineage_sizes[c])
            n_carriers = int(carrier_lineage_counts.get(c, 0))
            prevalence = n_carriers / n_strains_lineage if n_strains_lineage else float("nan")
            long_rows.append(dict(
                effector=eff, lineage=int(c),
                n_strains_lineage=n_strains_lineage,
                n_carriers=n_carriers,
                prevalence_in_lineage=round(prevalence, 4),
            ))
            share = n_carriers / total_carriers if total_carriers else 0.0
            per_lineage_shares.append(share)

        n_lineages_with_carrier = int((carrier_lineage_counts > 0).sum())
        max_lineage_share = round(max(per_lineage_shares), 4) if per_lineage_shares else float("nan")

        summary_rows.append(dict(
            effector=eff,
            total_carriers=total_carriers,
            n_lineages_with_carrier=n_lineages_with_carrier,
            max_lineage_share=max_lineage_share,
        ))

    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("max_lineage_share", ascending=False)

    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_LONG, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print(f"\n-> {OUT_LONG.relative_to(ROOT)}  ({len(long_df)} rows = 63 effectors x {len(lineages)} lineages)")
    print(f"-> {OUT_SUMMARY.relative_to(ROOT)}  ({len(summary_df)} rows = 63 effectors)")
    print(f"\ndone in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
