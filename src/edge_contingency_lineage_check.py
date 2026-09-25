#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_contingency_lineage_check.py
===================================
Independent, VAE-FREE robustness check for the 585 edges of
02_network_final_top30.json: does the raw co-occurrence/exclusion PATTERN
already visible in the 2x2 contingency table (the same table
negative_edges_analysis.py builds) survive leave-one-lineage-out exclusion?

This is conceptually distinct from edge_sign_stability_lineage.py (which
re-fits the Bernoulli-VAE on each lineage-excluded matrix and checks whether
the model's decoder-cosine weight keeps its sign): this script never touches
the VAE. It asks whether the DATA ITSELF -- the raw binary presence/absence
counts -- still shows the same directional tendency (co-occurrence vs.
mutual exclusion) after removing each lineage's strains, using nothing but
counting. The two checks are complementary: the VAE-based one validates
robustness of the REPORTED METHOD; this one validates whether the pattern is
already present in the raw data, independent of any model.

Reuses `contingency()` from negative_edges_analysis.py UNCHANGED (imported,
not duplicated) -- the exact same 2x2-table / lift / odds-ratio computation
already used for the original (non-lineage-excluded) negative-edge
validation, so both analyses are provably the same calculation applied to
different strain subsets.

Reuses the SAME strain -> lineage assignment as phylogenetic_sensitivity.py
and effector_lineage_distribution.py: reads
results/phylogenetic_sensitivity/06_lineage_clusters.csv directly, does not
recompute the genomic-distance clustering.

WHY NO FISHER / NO BH CORRECTION HERE (deliberate, not an oversight):
`contingency()` still runs scipy's one-sided Fisher exact test internally
(it is not modified) and its returned dict still contains a `p_value` key --
that field is simply not read or reported by this script. This analysis is
a DIRECTIONAL CONCORDANCE check (does lift stay on the same side of 1 / does
the log-odds keep the same sign as in the full network?), evaluated on 5
much smaller, non-independent, overlapping leave-one-out subsets of the same
150 strains. Running 585 x 5 = 2925 additional Fisher tests and BH-correcting
them here would misrepresent this as a new significance test when it is not
one -- the significance test already lives in negative_edges_analysis.py,
run once on the full 150-strain cohort over all 1953 candidate pairs.
This script only asks "same direction, yes/no".

Inputs (relative to the repository root):
    data/input/EffectorRepertoire_Ecoli.xlsx           -> binary matrix (Block 1 logic)
    results/02_network_final_top30.json                   -> the 585 reported edges
    results/phylogenetic_sensitivity/06_lineage_clusters.csv  -> strain -> lineage_cluster
Outputs:
    results/phylogenetic_sensitivity/09_edge_contingency_full_dataset_baseline.csv
        one row per edge, ALL 150 strains, no lineage excluded: n_both,
        n_a_only, n_b_only, n_neither, lift, odds_ratio, sign_raw_full_data,
        agrees_with_vae_sign
    results/phylogenetic_sensitivity/09_edge_contingency_stability_lineage.csv
        one row per (edge, excluded lineage): n_both, n_a_only, n_b_only,
        n_neither, lift, odds_ratio, direction_agrees_with_full_network
    results/phylogenetic_sensitivity/09_edge_contingency_stability_summary.csv
        one row per edge: n_lineage_exclusions_direction_agrees (0-5 of 5)

WHY THE FULL-DATASET BASELINE (09_edge_contingency_full_dataset_baseline.csv)
MATTERS, AND WHY IT LIVES HERE RATHER THAN AS A SEPARATE SCRIPT: it is the
same contingency computation as the lineage-exclusion loop below, just with
NO strain excluded, so it belongs next to that logic, not in a new module.
It answers a question the lineage-exclusion CSVs alone cannot: for the
edges where the raw contingency direction disagrees with the VAE's reported
sign under lineage exclusion (e.g. the n_lineage_exclusions_direction_agrees
= 0 edges in 09_edge_contingency_stability_summary.csv), did that disagreement
ALREADY EXIST with the complete, unexcluded 150-strain dataset? If yes, it
reflects a general difference between the VAE's score and simple marginal
co-occurrence (a modeling-choice difference, unrelated to lineage
structure) -- not evidence of fragility to population structure. If the
disagreement appears ONLY under some lineage exclusions but not in the
full-data baseline, THAT would indicate real phylogenetic sensitivity. This
baseline is the reference point needed to tell those two cases apart (see
09_edge_contingency_full_dataset_baseline.csv for the resulting breakdown).

Usage:
    python src/edge_contingency_lineage_check.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from inference_functions import load_binary_matrix  # noqa: E402
from negative_edges_analysis import contingency  # noqa: E402  (reused, not duplicated)

INPUT_XLSX = ROOT / "data" / "input" / "EffectorRepertoire_Ecoli.xlsx"
REF_JSON = ROOT / "results" / "02_network_final_top30.json"
LINEAGE_CSV = ROOT / "results" / "phylogenetic_sensitivity" / "06_lineage_clusters.csv"
OUT_LONG = ROOT / "results" / "phylogenetic_sensitivity" / "09_edge_contingency_stability_lineage.csv"
OUT_SUMMARY = ROOT / "results" / "phylogenetic_sensitivity" / "09_edge_contingency_stability_summary.csv"
OUT_BASELINE = ROOT / "results" / "phylogenetic_sensitivity" / "09_edge_contingency_full_dataset_baseline.csv"


def _direction(lift):
    """lift > 1 -> co-occurrence ('pos'); lift < 1 -> exclusion ('neg');
    lift == 1 or undefined (NaN, e.g. an effector has 0 carriers left after
    excluding this lineage) -> 'neutral', which by construction never
    agrees with either a 'pos' or 'neg' reference direction."""
    if not np.isfinite(lift):
        return "neutral"
    if lift > 1:
        return "pos"
    if lift < 1:
        return "neg"
    return "neutral"


def main() -> int:
    t0 = time.time()

    if not LINEAGE_CSV.is_file():
        raise FileNotFoundError(
            f"{LINEAGE_CSV} not found -- run phylogenetic_sensitivity.py first "
            "(this script reuses its strain -> lineage assignment)."
        )

    matrix = load_binary_matrix(INPUT_XLSX)
    print(f"matrix: {matrix.shape[0]} effectors x {matrix.shape[1]} strains")

    lineage_df = pd.read_csv(LINEAGE_CSV)
    strain_to_lineage = dict(zip(lineage_df["strain"], lineage_df["lineage_cluster"]))
    lineages = sorted(lineage_df["lineage_cluster"].unique())
    lineage_strains = {
        c: [s for s in matrix.columns if strain_to_lineage.get(s) == c]
        for c in lineages
    }
    print("lineages: " + ", ".join(f"C{c}(n={len(lineage_strains[c])})" for c in lineages))

    with open(REF_JSON, encoding="utf-8") as fh:
        net = json.load(fh)
    edges = net["elements"]["edges"]
    print(f"reference network: {len(edges)} edges (positive and negative)")

    # ---- full-dataset baseline (no lineage excluded) -----------------
    # Same contingency() call as the lineage-exclusion loop below, on the
    # complete 150-strain matrix -- see module docstring for why this is
    # needed to interpret the lineage-exclusion results correctly.
    baseline_rows = []
    for e in edges:
        a, b = e["data"]["source"], e["data"]["target"]
        w_ref = float(e["data"]["weight"])
        sign_ref = "pos" if w_ref >= 0 else "neg"
        ct = contingency(matrix, a, b)
        direction = _direction(ct["lift"])
        baseline_rows.append(dict(
            eff_a=a, eff_b=b, weight_ref=round(w_ref, 4), sign_ref=sign_ref,
            n_both=ct["n_both"], n_a_only=ct["n_a_only"],
            n_b_only=ct["n_b_only"], n_neither=ct["n_neither"],
            lift=round(ct["lift"], 4) if np.isfinite(ct["lift"]) else ct["lift"],
            odds_ratio=round(ct["odds_ratio"], 4) if np.isfinite(ct["odds_ratio"]) else ct["odds_ratio"],
            sign_raw_full_data=direction,
            agrees_with_vae_sign=bool(direction == sign_ref),
        ))
    baseline_df = pd.DataFrame(baseline_rows)
    OUT_BASELINE.parent.mkdir(parents=True, exist_ok=True)
    baseline_df.to_csv(OUT_BASELINE, index=False)

    n_base_agree = int(baseline_df["agrees_with_vae_sign"].sum())
    print(f"\nfull-dataset baseline (no lineage excluded): VAE sign vs. raw lift/odds-ratio "
          f"agree on {n_base_agree}/{len(baseline_df)} edges "
          f"({100 * n_base_agree / len(baseline_df):.2f}%)")
    print(f"-> {OUT_BASELINE.relative_to(ROOT)}")

    long_rows = []
    for e in edges:
        a, b = e["data"]["source"], e["data"]["target"]
        w_ref = float(e["data"]["weight"])
        sign_ref = "pos" if w_ref >= 0 else "neg"

        for c in lineages:
            excl_strains = lineage_strains[c]
            sub = matrix.drop(columns=excl_strains)
            ct = contingency(sub, a, b)
            direction = _direction(ct["lift"])
            agrees = direction == sign_ref

            long_rows.append(dict(
                eff_a=a, eff_b=b, weight_ref=round(w_ref, 4), sign_ref=sign_ref,
                excluded_lineage=int(c), n_strains_excluded=len(excl_strains),
                n_both=ct["n_both"], n_a_only=ct["n_a_only"],
                n_b_only=ct["n_b_only"], n_neither=ct["n_neither"],
                lift=round(ct["lift"], 4) if np.isfinite(ct["lift"]) else ct["lift"],
                odds_ratio=round(ct["odds_ratio"], 4) if np.isfinite(ct["odds_ratio"]) else ct["odds_ratio"],
                direction=direction,
                direction_agrees_with_full_network=bool(agrees),
            ))

    long_df = pd.DataFrame(long_rows)
    OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_LONG, index=False)

    summary_rows = []
    for (a, b), grp in long_df.groupby(["eff_a", "eff_b"], sort=False):
        summary_rows.append(dict(
            eff_a=a, eff_b=b,
            weight_ref=grp["weight_ref"].iloc[0], sign_ref=grp["sign_ref"].iloc[0],
            n_lineage_exclusions_direction_agrees=int(grp["direction_agrees_with_full_network"].sum()),
            n_lineage_exclusions_total=len(grp),
        ))
    # preserve the edges' original order from the reference JSON
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    n_total = len(long_df)
    n_agree = int(long_df["direction_agrees_with_full_network"].sum())
    print("\n" + "=" * 70)
    print("  SUMMARY -- directional concordance under leave-one-lineage-out")
    print("=" * 70)
    print(f"  (edge, excluded-lineage) pairs checked : {n_total} (585 edges x 5 lineages)")
    print(f"  direction agrees with full network     : {n_agree}/{n_total} "
          f"({100 * n_agree / n_total:.2f}%)")

    full5 = int((summary_df["n_lineage_exclusions_direction_agrees"] == 5).sum())
    print(f"  edges agreeing in ALL 5 exclusions      : {full5}/{len(summary_df)} "
          f"({100 * full5 / len(summary_df):.2f}%)")

    # ---- baseline cross-check for the edges that disagree in ALL 5
    # lineage exclusions (n_lineage_exclusions_direction_agrees == 0) --
    # were they already disagreeing in the unexcluded, full-data baseline?
    zero5 = summary_df[summary_df["n_lineage_exclusions_direction_agrees"] == 0]
    merged0 = zero5.merge(baseline_df[["eff_a", "eff_b", "agrees_with_vae_sign"]],
                          on=["eff_a", "eff_b"], how="left")
    n_zero5 = len(merged0)
    n_zero5_already_in_baseline = int((~merged0["agrees_with_vae_sign"]).sum())
    print(f"\n  edges disagreeing in ALL 5 lineage exclusions (0/5) : {n_zero5}")
    if n_zero5:
        print(f"    of those, already disagreeing in the full-dataset baseline "
              f"(no exclusion) : {n_zero5_already_in_baseline}/{n_zero5} "
              f"({100 * n_zero5_already_in_baseline / n_zero5:.2f}%)")
        print(f"    -> disagreement pre-existing in the full data, NOT induced by "
              f"lineage exclusion, for {n_zero5_already_in_baseline}/{n_zero5} of "
              f"these edges.")

    print(f"\n  -> {OUT_LONG.relative_to(ROOT)}")
    print(f"  -> {OUT_SUMMARY.relative_to(ROOT)}")
    print(f"\n  done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
