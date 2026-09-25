#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mantel-Haenszel phylogeny-corrected pairwise association + bootstrap-supported
hierarchical clustering (Reviewer 2, point 3, Methods 2.9).

REPORTED METHOD: Mantel-Haenszel stratified log-odds is the ONLY method used
for Reviewer 2's point 3 in the manuscript. By default this script computes:
(1) the 5-lineage UPGMA strain partition already computed and validated
against the NJ tree in phylogenetic_sensitivity.py's phase1()
(results/phylogenetic_sensitivity/06_lineage_clusters.csv, 75/52/9/7/7 strains,
reused here, not re-derived -- Reviewer 2's "do not fix the number of
clusters a priori" requirement applies to Step 3 below, i.e. the number of
effector modules on the corrected association matrix, not to how many
strain lineages Step 1 uses as strata), (2) the Mantel-Haenszel log-odds
association matrix, computed WITHIN lineages and pooled ACROSS lineages,
(3) bootstrap-supported hierarchical clustering of that matrix (k chosen by
silhouette, not fixed), and (4) the adjusted Rand index of that clustering
against the reported VAE network's 7 modules. COMPLEMENTARY to the
leave-one-lineage-out VAE sensitivity analysis (phylogenetic_sensitivity.py,
Reviewer 1) -- this script does NOT touch the VAE anywhere; it is a
deliberately independent, classical, phylogeny-aware check.

L1-LOGISTIC DIAGNOSTIC (--include-l1-diagnostic): an L1-logistic-bagged
(inverse-variance pooled) alternative to Mantel-Haenszel. It collapses to
~0 for nearly every pair once fit within lineages, even in the 75-strain
lineage (unlike its well-behaved non-stratified use in Methods 2.5), so it
carries no usable signal (ARI vs the VAE modules ~0.02-0.03 regardless of
lineage partition; bootstrap cluster support ~0). This is diagnostic
evidence for why Mantel-Haenszel was chosen (Table S12), not a reported
association/clustering result -- its outputs are named with an EXCLUDED
marker so nobody mistakes them for the reported tables. See
step2_l1_diagnostic()'s docstring for the one-line reason.

Inputs (relative to the repository root):
    data/genomes/ecoli_150_distance_matrix.csv   (dendrogram visual only)
    data/input/EffectorRepertoire_Ecoli.xlsx
    results/phylogenetic_sensitivity/06_lineage_clusters.csv   (5-lineage UPGMA
        partition, Step 1 -- reused, not re-derived)
    results/TableS9.csv   (VAE network's 7 modules, Step 4 only)

Outputs (results/phylo_corrected_convergence/), default run:
    12_lineage_assignment.csv
    figures/12_lineage_dendrogram_qc.{pdf,png}
    12_association_MH_logodds_phylo_corrected.csv    <- REPORTED (Table/Methods 2.9)
    12_clusters_MH_composition_support.csv           <- REPORTED (Table/Methods 2.9)
    12_ari_permutation_test.csv                       <- REPORTED (Methods 2.9): label-permutation
                                                       null test for the ARI (Step 5, n=10000,
                                                       VAE modules fixed, MH labels shuffled)

Additional outputs with --include-l1-diagnostic (Table S12):
    12_l1_stratified_diagnostic_EXCLUDED_association.csv
    12_l1_stratified_diagnostic_EXCLUDED_clusters.csv

Usage:
    python src/phylo_corrected_convergence.py [--n-boot-stability 500] [--include-l1-diagnostic]

Wired into run_all.py as step 12, with --n-boot-stability 500
--include-l1-diagnostic (the reported bootstrap count, with the diagnostic
included so Table S12 is reproduced). The Step-3 bootstrap (pvclust-style
cluster support) defaults to B=20 here for a quick standalone smoke test;
the number reported in the manuscript is B=500.
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import os as _os
_os.environ.setdefault("MPLBACKEND", "Agg")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, adjusted_rand_score

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from inference_functions import load_binary_matrix, bagged_l1_network, neighborhood_selection_logistic  # noqa: E402
from negative_edges_analysis import contingency  # noqa: E402
from figure_style import *  # noqa: E402,F401,F403

init_style()

# ── configuration ────────────────────────────────────────────────────────────
DIST_CSV = ROOT / "data" / "genomes" / "ecoli_150_distance_matrix.csv"
INPUT_XLSX = ROOT / "data" / "input" / "EffectorRepertoire_Ecoli.xlsx"
VAE_CLUSTERING_CSV = ROOT / "results" / "TableS9.csv"
LINEAGE_CSV = ROOT / "results" / "phylogenetic_sensitivity" / "06_lineage_clusters.csv"
OUT_DIR = ROOT / "results" / "phylo_corrected_convergence"
FIG_DIR = OUT_DIR / "figures"

L1_N_BOOTSTRAP = 30            # same as run_phase2c_l1_bagged
L1_SAMPLE_FRACTION = 0.6       # same as run_phase2c_l1_bagged
SEED = 42
VAR_EPS = 1e-6                 # floor added to per-lineage bootstrap variance before inverting
HALDANE = 0.5                  # continuity correction for zero cells in the MH 2x2 tables
MAX_K = 15                     # silhouette sweep upper bound (same range used for the VAE modules)
TIE_TOL = 1e-3                 # silhouette values within this of the max are treated as a tie
JACCARD_SUPPORT_THRESHOLD = 0.7  # "same or very similar cluster" rule for bootstrap support
N_PERMUTATIONS = 10000         # Step 5: label-permutation null test for the ARI

DENDRO_PALETTE = [
    "#E8A0C8", "#C85A9A", "#90C4F4", "#4A96D8", "#9A90D8",
    "#5A4AAE", "#6ACFB0", "#2A9E80", "#D4A080", "#B06A40",
]
# ─────────────────────────────────────────────────────────────────────────────


def strip_version(acc):
    return str(acc).split(".")[0]


def silhouette_k_sweep(dist, max_k=MAX_K, linkage_method="average"):
    """Average-linkage clustering on a precomputed distance matrix, sweeping
    k = 2..max_k and scoring each with the silhouette coefficient."""
    n = dist.shape[0]
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    scores = {}
    for k in range(2, min(max_k, n - 1) + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        scores[k] = silhouette_score(dist, labels, metric="precomputed")
    return Z, scores


def choose_k(scores, tol=TIE_TOL):
    """argmax silhouette; k's within `tol` of the max are a near-tie, broken
    by parsimony (lowest k among the tied set) -- documented explicitly by
    the caller, never applied silently."""
    best = max(scores.values())
    near_ties = sorted(k for k, v in scores.items() if best - v <= tol)
    return near_ties[0], near_ties, best


def association_to_distance(assoc_df):
    """Signed association matrix -> distance matrix for clustering: scale by
    the 99th percentile of |non-zero values| (clip to [-1, 1]) so both the L1
    and the (unbounded) MH log-odds matrices land on a comparable scale,
    then d = 1 - sim, symmetrized, diagonal zeroed, negative residue clipped."""
    vals = assoc_df.values.astype(float).copy()
    iu = np.triu_indices_from(vals, k=1)
    abs_nonzero = np.abs(vals[iu])
    abs_nonzero = abs_nonzero[np.isfinite(abs_nonzero) & (abs_nonzero > 0)]
    scale = float(np.percentile(abs_nonzero, 99)) if len(abs_nonzero) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    vals = np.nan_to_num(vals, nan=0.0, posinf=scale, neginf=-scale)
    sim = np.clip(vals / scale, -1.0, 1.0)
    dist = 1.0 - sim
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, None)
    return dist, scale


def cluster_solution(dist, effectors, max_k=MAX_K, linkage_method="average"):
    Z, scores = silhouette_k_sweep(dist, max_k=max_k, linkage_method=linkage_method)
    k, ties, best = choose_k(scores)
    labels = fcluster(Z, t=k, criterion="maxclust")
    clusters = {}
    for eff, lab in zip(effectors, labels):
        clusters.setdefault(int(lab), []).append(eff)
    cluster_map = dict(zip(effectors, (int(x) for x in labels)))
    return Z, scores, k, ties, clusters, cluster_map


# ── Step 1 ───────────────────────────────────────────────────────────────────
def step1_lineages():
    """
    Reuses the 5-lineage UPGMA partition already computed and validated
    against the NJ tree in phylogenetic_sensitivity.py's phase1()
    (LINEAGE_CSV) -- does NOT re-derive a partition by its own silhouette
    sweep (see module docstring CHANGE note for why). The genomic distance
    matrix is still loaded, purely to draw the dendrogram (a visual QC
    figure colored by the reused lineages), not to pick the partition.
    """
    print("\n" + "=" * 70)
    print("STEP 1 -- lineage assignment (reusing the published 5-lineage UPGMA partition)")
    print("=" * 70)
    D = pd.read_csv(DIST_CSV, index_col=0, float_precision="round_trip")
    labels_full = list(D.index)
    labels = [strip_version(x) for x in labels_full]
    A = D.to_numpy(dtype=float)
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A[A < 0] = 0.0
    n = len(labels)

    lin_df = pd.read_csv(LINEAGE_CSV)
    final_map = {row["strain"]: int(row["lineage_cluster"]) for _, row in lin_df.iterrows()}
    missing = [s for s in labels if s not in final_map]
    if missing:
        raise ValueError(
            f"{len(missing)} strain(s) in {DIST_CSV.name} are missing from "
            f"{LINEAGE_CSV.relative_to(ROOT)}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    final_clusters = {}
    for s in labels:
        final_clusters.setdefault(final_map[s], []).append(s)
    final_clusters = {cid: sorted(members) for cid, members in sorted(final_clusters.items())}

    print(f"  source: {LINEAGE_CSV.relative_to(ROOT)} "
          f"(same 5-lineage UPGMA partition as phylogenetic_sensitivity.py's phase1(), "
          f"validated against the NJ tree)")
    print("\n  lineages:")
    for cid, members in final_clusters.items():
        print(f"    lineage {cid}: n={len(members)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    lineage_df = pd.DataFrame({
        "strain": labels, "accession_full": labels_full,
        "lineage": [final_map[s] for s in labels],
    }).sort_values(["lineage", "strain"])
    lineage_df.to_csv(OUT_DIR / "12_lineage_assignment.csv", index=False)

    # dendrogram: visual QC only, average-linkage on the same genomic distance
    # matrix, colored by the reused lineage partition above (not used to
    # derive it).
    condensed = squareform(A, checks=False)
    Z = linkage(condensed, method="average")

    color_map = {cid: DENDRO_PALETTE[(cid - 1) % len(DENDRO_PALETTE)] for cid in final_clusters}
    link_members = {}
    for i, row in enumerate(Z):
        left, right = int(row[0]), int(row[1])
        node_id = n + i
        left_members = {left} if left < n else link_members[left]
        right_members = {right} if right < n else link_members[right]
        link_members[node_id] = left_members | right_members

    def link_color(node_id):
        member_lineages = {final_map[labels[i]] for i in link_members[node_id]}
        if len(member_lineages) == 1:
            return color_map[member_lineages.pop()]
        return "#B8B8B8"

    fig, ax = plt.subplots(figsize=figsize(W_FULL, H_WIDE))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=4,
               link_color_func=link_color, above_threshold_color="#B8B8B8", ax=ax)
    style(ax, xlabel="Strain", ylabel="Genomic distance (1 - ANI)")
    save(fig, str(FIG_DIR / "12_lineage_dendrogram_qc"), fmt="pdf")
    save(fig, str(FIG_DIR / "12_lineage_dendrogram_qc"), fmt="png")
    plt.close(fig)

    print(f"\n  -> {OUT_DIR.relative_to(ROOT)}/12_lineage_assignment.csv, "
          f"figures/12_lineage_dendrogram_qc.{{pdf,png}}")
    return labels, final_map, final_clusters


# ── Step 2 ───────────────────────────────────────────────────────────────────
def mh_log_odds(matrix, effectors, lineage_strains, haldane=HALDANE):
    """Mantel-Haenszel common log-odds-ratio, stratified by lineage.

    A stratum in which EITHER effector is monomorphic within that lineage
    (present or absent in 100% of that lineage's strains -- equivalently,
    the 2x2 table's row margin or column margin is 0 or n_l) carries no
    information about the pairwise association -- that effector has zero
    within-lineage variance, so no within-stratum co-occurrence signal is
    possible. Such degenerate strata are excluded entirely (no Haldane
    continuity correction, no raw contribution) for that pair: applying the
    Haldane correction (+0.5 to all 4 cells) to a degenerate stratum would
    manufacture a nonzero, direction-dependent contribution to num_or/den_or
    purely from the correction constant, not from data, diluting/distorting
    the pooled estimate -- whereas an uncorrected degenerate stratum already
    contributes exactly 0 to both num_or and den_or (one of a*d, b*c is
    always 0), so exclusion is the correct classical MH treatment of
    non-informative strata. Haldane is applied only to strata where BOTH
    effectors vary within the lineage (genuinely informative) but still have
    a small-sample zero cell -- the standard use of the correction.
    If ALL 5 strata are degenerate for a pair (no lineage is informative),
    the pair is set to NaN (excluded downstream by association_to_distance,
    which already treats NaN as neutral/zero similarity).
    """
    n_eff = len(effectors)
    out = np.zeros((n_eff, n_eff))
    n_all_degenerate = 0
    for ia, ib in combinations(range(n_eff), 2):
        eff_a, eff_b = effectors[ia], effectors[ib]
        num_or, den_or = 0.0, 0.0
        n_informative = 0
        for strains in lineage_strains.values():
            sub = matrix[strains]
            ct = contingency(sub, eff_a, eff_b)
            a, b, c, d = ct["n_both"], ct["n_a_only"], ct["n_b_only"], ct["n_neither"]
            nl = a + b + c + d
            if nl == 0:
                continue
            row_margin = a + b  # count with eff_a present, regardless of eff_b
            col_margin = a + c  # count with eff_b present, regardless of eff_a
            if row_margin in (0, nl) or col_margin in (0, nl):
                continue  # eff_a or eff_b monomorphic in this lineage -- not informative, excluded
            n_informative += 1
            a_, b_, c_, d_ = a + haldane, b + haldane, c + haldane, d + haldane
            num_or += (a_ * d_) / nl
            den_or += (b_ * c_) / nl
        if n_informative == 0:
            n_all_degenerate += 1
            out[ia, ib] = out[ib, ia] = np.nan
            continue
        lo = np.log(num_or / den_or) if den_or > 0 and num_or > 0 else np.nan
        out[ia, ib] = out[ib, ia] = lo
    if n_all_degenerate:
        print(f"  [mh_log_odds] {n_all_degenerate} pair(s) had ALL 5 lineages degenerate "
              f"(no informative stratum) -- set to NaN")
    return out


def naive_log_odds(matrix, effectors, haldane=HALDANE):
    n_eff = len(effectors)
    out = np.zeros((n_eff, n_eff))
    for ia, ib in combinations(range(n_eff), 2):
        eff_a, eff_b = effectors[ia], effectors[ib]
        ct = contingency(matrix, eff_a, eff_b)
        a = ct["n_both"] + haldane
        b = ct["n_a_only"] + haldane
        c = ct["n_b_only"] + haldane
        d = ct["n_neither"] + haldane
        out[ia, ib] = out[ib, ia] = np.log((a * d) / (b * c))
    return out


def build_lineage_strains(matrix, final_clusters):
    lineage_strains = {c: [s for s in members if s in matrix.columns]
                        for c, members in final_clusters.items()}
    for c, members in final_clusters.items():
        missing = len(members) - len(lineage_strains[c])
        if missing:
            print(f"  WARNING: lineage {c}: {missing} strain(s) in the distance matrix "
                  f"not found among the binary matrix's columns")
    return lineage_strains


def step2_mh(matrix, effectors, lineage_strains):
    """Mantel-Haenszel stratified log-odds -- the REPORTED method (Methods 2.9)."""
    print("\n" + "=" * 70)
    print("STEP 2 -- Mantel-Haenszel stratified log-odds (phylogeny-corrected association)")
    print("=" * 70)

    print("\n  Mantel-Haenszel stratified log-odds (contingency() per lineage)")
    t0 = time.time()
    mh_arr = mh_log_odds(matrix, effectors, lineage_strains)
    naive_lo_arr = naive_log_odds(matrix, effectors)
    print(f"    {len(effectors) * (len(effectors)-1)//2} pairs x {len(lineage_strains)} "
          f"lineages: {time.time()-t0:.1f} s")
    mh_df = pd.DataFrame(mh_arr, index=effectors, columns=effectors)
    naive_lo_df = pd.DataFrame(naive_lo_arr, index=effectors, columns=effectors)

    rng = np.random.default_rng(SEED)
    pairs = list(combinations(effectors, 2))
    sample_idx = rng.choice(len(pairs), size=min(8, len(pairs)), replace=False)

    print("\n  sample pairs -- log-odds (naive, pooled vs Mantel-Haenszel corrected):")
    print(f"    {'pair':<32} {'naive':>9} {'MH':>10}")
    for idx in sample_idx:
        a, b = pairs[idx]
        print(f"    {a + '-' + b:<32} {naive_lo_df.loc[a, b]:>9.4f} {mh_df.loc[a, b]:>10.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mh_df.to_csv(OUT_DIR / "12_association_MH_logodds_phylo_corrected.csv")
    print(f"\n  -> {OUT_DIR.relative_to(ROOT)}/12_association_MH_logodds_phylo_corrected.csv")

    return mh_df


def step2_l1_diagnostic(matrix, effectors, lineage_strains):
    """
    DIAGNOSTIC ONLY, not reported in the manuscript: L1-logistic bagged
    (inverse-variance pooled across lineages) collapses to ~0 for nearly
    every pair once fit within lineages -- confirmed even in the 75-strain
    lineage, so it is not a small-sample artefact -- unlike its well-behaved
    non-stratified use in Methods 2.5. Kept here, opt-in only, as evidence
    for why Mantel-Haenszel (step2_mh) is the method actually used for
    Reviewer 2's point 3.
    """
    print("\n" + "=" * 70)
    print("STEP 2 [DIAGNOSTIC, --include-l1-diagnostic] -- L1 logistic bagged, EXCLUDED from the manuscript")
    print("=" * 70)

    print(f"\n  L1 logistic bagged per lineage "
          f"(n_bootstrap={L1_N_BOOTSTRAP}, sample_fraction={L1_SAMPLE_FRACTION})")
    lineage_means, lineage_vars = {}, {}
    for c, strains in lineage_strains.items():
        sub = matrix[strains]
        t0 = time.time()
        consensus_df, replicates = bagged_l1_network(
            sub, n_bootstrap=L1_N_BOOTSTRAP, sample_fraction=L1_SAMPLE_FRACTION,
            seed=SEED, n_jobs=-2,
        )
        stack = np.stack([r.values for r in replicates], axis=0)
        lineage_means[c] = consensus_df.values
        lineage_vars[c] = stack.var(axis=0, ddof=1)
        print(f"    lineage {c} (n={len(strains)} strains): {(time.time()-t0)/60:.2f} min")

    weights = {c: 1.0 / (lineage_vars[c] + VAR_EPS) for c in lineage_means}
    num = sum(weights[c] * lineage_means[c] for c in lineage_means)
    den = sum(weights[c] for c in lineage_means)
    l1_corrected_df = pd.DataFrame(num / den, index=effectors, columns=effectors)
    print("\n  pooling formula (inverse-variance, Cochran-Mantel-Haenszel principle "
          "applied to per-lineage point estimates instead of 2x2 tables):")
    print("    corrected[i,j] = sum_l w_l[i,j]*mean_l[i,j] / sum_l w_l[i,j], "
          f"w_l[i,j] = 1 / (var_l[i,j] + {VAR_EPS:g})")
    print(f"    var_l[i,j] = variance across the {L1_N_BOOTSTRAP} within-lineage bootstrap "
          f"replicates already fit above (no extra work)")

    t0 = time.time()
    naive_l1_df, _ = bagged_l1_network(matrix, n_bootstrap=L1_N_BOOTSTRAP,
                                        sample_fraction=L1_SAMPLE_FRACTION, seed=SEED, n_jobs=-2)
    print(f"\n  naive (non-stratified, all 150 strains) L1 bagged: {(time.time()-t0)/60:.2f} min")

    rng = np.random.default_rng(SEED)
    pairs = list(combinations(effectors, 2))
    sample_idx = rng.choice(len(pairs), size=min(8, len(pairs)), replace=False)

    print("\n  sample pairs -- L1 (naive, all 150 strains vs phylogeny-corrected):")
    print(f"    {'pair':<32} {'naive':>9} {'corrected':>10}")
    for idx in sample_idx:
        a, b = pairs[idx]
        print(f"    {a + '-' + b:<32} {naive_l1_df.loc[a, b]:>9.4f} {l1_corrected_df.loc[a, b]:>10.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "12_l1_stratified_diagnostic_EXCLUDED_association.csv"
    l1_corrected_df.to_csv(out_path)
    print(f"\n  -> {out_path.relative_to(ROOT)}  (DIAGNOSTIC -- not a reported result)")

    return l1_corrected_df, weights


# ── Step 3 ───────────────────────────────────────────────────────────────────
def bootstrap_replicate_l1(seed, matrix, lineage_strains, weights, effectors):
    """One outer bootstrap resample: resample each lineage's OWN strains with
    replacement (stratified, preserves per-lineage size and the CMH-style
    stratification), fit ONE (not bagged) L1-logistic per lineage, and pool
    with the FIXED inverse-variance weights from Step 2a -- reusing those
    weights, rather than recomputing 30 internal bootstrap replicates per
    outer resample, is what keeps this step computationally tractable."""
    rng = np.random.default_rng(seed)
    lineage_est = {}
    for c, strains in lineage_strains.items():
        idx = rng.choice(len(strains), size=len(strains), replace=True)
        sub = matrix[[strains[i] for i in idx]]
        adj_df, _ = neighborhood_selection_logistic(sub, normalize_adj=False, symm_rule="average_signed")
        lineage_est[c] = adj_df.values
    num = sum(weights[c] * lineage_est[c] for c in lineage_est)
    den = sum(weights[c] for c in lineage_est)
    return num / den


def bootstrap_replicate_mh(seed, matrix, lineage_strains, effectors):
    rng = np.random.default_rng(seed)
    boot_strains = {}
    for c, strains in lineage_strains.items():
        idx = rng.choice(len(strains), size=len(strains), replace=True)
        boot_strains[c] = [strains[i] for i in idx]
    return mh_log_odds(matrix, effectors, boot_strains)


def bootstrap_stability(method_name, effectors, orig_clusters, matrix, lineage_strains,
                         weights, B, jaccard_threshold=JACCARD_SUPPORT_THRESHOLD, seed=SEED):
    print(f"\n  [{method_name}] bootstrap cluster support: B={B} resamples "
          f"(stratified resampling of each lineage's own strains, single fit per lineage)")
    t0 = time.time()

    def one_replicate(b):
        if method_name == "L1":
            corrected = bootstrap_replicate_l1(seed + b, matrix, lineage_strains, weights, effectors)
        else:
            corrected = bootstrap_replicate_mh(seed + b, matrix, lineage_strains, effectors)
        dist, _ = association_to_distance(pd.DataFrame(corrected, index=effectors, columns=effectors))
        _, _, _, _, clusters, _ = cluster_solution(dist, effectors)
        return clusters

    replicate_clusters = Parallel(n_jobs=-2)(delayed(one_replicate)(b) for b in range(B))
    dt = time.time() - t0
    print(f"    done in {dt:.1f} s ({dt / max(B, 1):.2f} s/replicate) "
          f"-> extrapolated to B=500: {dt / max(B, 1) * 500 / 60:.1f} min")

    support = {}
    for cid, members in orig_clusters.items():
        s_orig = set(members)
        best_per_rep = []
        for rep_clusters in replicate_clusters:
            best = max(
                (len(s_orig & set(m)) / len(s_orig | set(m)) for m in rep_clusters.values()),
                default=0.0,
            )
            best_per_rep.append(best)
        support[cid] = float(np.mean([s >= jaccard_threshold for s in best_per_rep]))
    return support, dt


def step3_clustering(method_configs, effectors, matrix, lineage_strains, n_boot):
    """method_configs: list of (name, assoc_df, weights, out_path, label_suffix)."""
    print("\n" + "=" * 70)
    print("STEP 3 -- bootstrap-supported hierarchical clustering")
    print("=" * 70)
    print(f"  cluster-support rule: for each original cluster, the fraction of bootstrap "
          f"resamples in which some cluster of the resampled solution has Jaccard overlap "
          f">= {JACCARD_SUPPORT_THRESHOLD} with it (a pvclust-style support proportion, "
          f"not the formal multiscale-bootstrap AU p-value pvclust reports).")

    results = {}
    for name, assoc_df, weights, out_path, label_suffix in method_configs:
        dist, scale = association_to_distance(assoc_df)
        Z, scores, k, ties, clusters, cluster_map = cluster_solution(dist, effectors)
        print(f"\n  [{name}{label_suffix}] distance scale (99th pct |assoc|) = {scale:.4f}")
        print(f"  [{name}{label_suffix}] silhouette-selected k = {k}" +
              (f"  (near-ties at k in {ties}, parsimony applied)" if len(ties) > 1 else ""))
        for cid, members in clusters.items():
            print(f"    cluster {cid} (n={len(members)}): {', '.join(members)}")

        support, dt = bootstrap_stability(name, effectors, clusters, matrix, lineage_strains,
                                           weights, B=n_boot)
        print(f"\n  [{name}{label_suffix}] bootstrap support (B={n_boot}):")
        for cid in clusters:
            print(f"    cluster {cid}: support = {support[cid]:.2f}")

        comp_rows = [{"cluster": cid, "effector": e, "bootstrap_support": support[cid]}
                     for cid, members in clusters.items() for e in members]
        pd.DataFrame(comp_rows).to_csv(out_path, index=False)
        print(f"  -> {out_path.relative_to(ROOT)}{label_suffix}")

        results[name] = {"clusters": clusters, "cluster_map": cluster_map, "support": support,
                          "k": k, "dt_boot": dt}
    return results


# ── Step 4 ───────────────────────────────────────────────────────────────────
def step4_ari(results, effectors):
    print("\n" + "=" * 70)
    print("STEP 4 -- agreement with the VAE network's modules (adjusted Rand index)")
    print("=" * 70)
    vae_df = pd.read_csv(VAE_CLUSTERING_CSV)
    vae_map = dict(zip(vae_df["effector"], vae_df["cluster"]))
    missing = [e for e in effectors if e not in vae_map]
    if missing:
        print(f"  WARNING: {len(missing)} effectors missing from {VAE_CLUSTERING_CSV.name}: {missing}")
    common = [e for e in effectors if e in vae_map]
    vae_labels = [vae_map[e] for e in common]

    aris = {}
    for name in results:
        labels = [results[name]["cluster_map"][e] for e in common]
        ari = adjusted_rand_score(vae_labels, labels)
        aris[name] = ari
        print(f"  ARI({name} phylo-corrected clusters, VAE {vae_df['cluster'].nunique()} modules) "
              f"= {ari:.4f}  (n={len(common)} effectors)")
    return aris


# ── Step 5 ───────────────────────────────────────────────────────────────────
def step5_ari_permutation(results, effectors, n_perm=N_PERMUTATIONS, seed=SEED):
    """
    Label-permutation null test for each method's ARI (Step 4) against the
    VAE's 7 modules: is the observed ARI higher than expected by chance,
    given how few of the phylogeny-corrected clusters have high bootstrap
    support?

    Convention: the VAE module partition is held FIXED (it is the established
    reference network being compared against) and the phylogeny-corrected
    partition's labels are shuffled uniformly at random across the 63
    effectors, preserving its own cluster sizes exactly. This is
    mathematically equivalent to permuting the VAE side instead -- ARI's
    null distribution under a fixed-size random relabeling is exchangeable
    between the two partitions being compared -- but fixing the established
    reference and permuting the partition under test is the more common
    convention and the more natural framing of the question being asked
    here ("is MH's data-driven clustering more concordant with the known VAE
    modules than chance would produce").
    """
    print("\n" + "=" * 70)
    print("STEP 5 -- label-permutation null test for the ARI")
    print("=" * 70)
    vae_df = pd.read_csv(VAE_CLUSTERING_CSV)
    vae_map = dict(zip(vae_df["effector"], vae_df["cluster"]))
    common = [e for e in effectors if e in vae_map]
    vae_labels = np.array([vae_map[e] for e in common])

    rows = []
    for name in results:
        method_labels = np.array([results[name]["cluster_map"][e] for e in common])
        observed_ari = adjusted_rand_score(vae_labels, method_labels)

        t0 = time.time()
        rng = np.random.default_rng(seed)
        perm_aris = np.empty(n_perm)
        for i in range(n_perm):
            perm_aris[i] = adjusted_rand_score(vae_labels, rng.permutation(method_labels))
        dt = time.time() - t0

        perm_mean = float(perm_aris.mean())
        perm_std = float(perm_aris.std(ddof=1))
        p_empirical = float(np.mean(perm_aris >= observed_ari))

        print(f"\n  [{name}] observed ARI = {observed_ari:.4f}")
        print(f"  [{name}] permutation null (n={n_perm}, {dt:.1f} s): "
              f"mean={perm_mean:.4f}  sd={perm_std:.4f}")
        print(f"  [{name}] empirical p-value (P[perm ARI >= observed]) = {p_empirical:.4g}")

        rows.append(dict(
            method=name, n_effectors=len(common), observed_ari=observed_ari,
            n_permutations=n_perm, perm_ari_mean=perm_mean, perm_ari_std=perm_std,
            p_empirical=p_empirical,
        ))

    out_path = OUT_DIR / "12_ari_permutation_test.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n  -> {out_path.relative_to(ROOT)}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot-stability", type=int, default=20,
                     help="B for the Step-3 bootstrap cluster-support test (default 20, a timing "
                          "probe -- rerun with 500 once the timing has been reviewed)")
    ap.add_argument("--include-l1-diagnostic", action="store_true",
                     help="Also compute the L1-logistic-bagged diagnostic (Steps 2/3/4 for L1). "
                          "NOT part of the manuscript's reported analysis -- kept only as "
                          "evidence for why Mantel-Haenszel was used instead. Off by default; "
                          "never called by run_all.py.")
    args = ap.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("  MANTEL-HAENSZEL PHYLOGENY-CORRECTED PAIRWISE ASSOCIATION + BOOTSTRAP CLUSTERING")
    print("  (Reviewer 2, point 3, Methods 2.9 -- complementary to the VAE-based LOLO analysis)")
    print("=" * 70)
    if args.include_l1_diagnostic:
        print("  --include-l1-diagnostic set: also running the EXCLUDED L1 diagnostic (not reported)")

    matrix = load_binary_matrix(INPUT_XLSX)
    effectors = matrix.index.tolist()
    print(f"  binary matrix: {matrix.shape[0]} effectors x {matrix.shape[1]} strains")

    t0 = time.time()
    _, _, final_clusters = step1_lineages()
    print(f"\n[timing] step 1: {(time.time() - t0) / 60:.2f} min")

    lineage_strains = build_lineage_strains(matrix, final_clusters)

    t0 = time.time()
    mh_df = step2_mh(matrix, effectors, lineage_strains)
    method_configs = [("MH", mh_df, None,
                        OUT_DIR / "12_clusters_MH_composition_support.csv", "")]
    if args.include_l1_diagnostic:
        l1_df, weights = step2_l1_diagnostic(matrix, effectors, lineage_strains)
        method_configs.append(("L1", l1_df, weights,
                                OUT_DIR / "12_l1_stratified_diagnostic_EXCLUDED_clusters.csv",
                                " DIAGNOSTIC, EXCLUDED"))
    print(f"\n[timing] step 2: {(time.time() - t0) / 60:.2f} min")

    t0 = time.time()
    results = step3_clustering(method_configs, effectors, matrix, lineage_strains,
                                n_boot=args.n_boot_stability)
    print(f"\n[timing] step 3 (B={args.n_boot_stability} probe): {(time.time() - t0) / 60:.2f} min")

    t0 = time.time()
    step4_ari(results, effectors)
    print(f"\n[timing] step 4: {(time.time() - t0) / 60:.2f} min")

    t0 = time.time()
    step5_ari_permutation(results, effectors)
    print(f"\n[timing] step 5: {(time.time() - t0) / 60:.2f} min")

    print(f"\n[timing] TOTAL: {(time.time() - t_start) / 60:.2f} min")
    print("\n" + "=" * 70)
    if not args.include_l1_diagnostic:
        print("  NOTE: the L1-logistic diagnostic was skipped (default). It is not part of the "
              "reported analysis -- rerun with --include-l1-diagnostic only if you need to "
              "re-examine why it was discarded.")
    if args.n_boot_stability < 500:
        print(f"  NOTE: Step 3 ran with B={args.n_boot_stability} as a timing probe, per the "
              f"analysis plan -- extrapolated B=500 timings are printed above per method. "
              f"Review those before rerunning with --n-boot-stability 500.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
