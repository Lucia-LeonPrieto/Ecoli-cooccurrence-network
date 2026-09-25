#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phylogenetic sensitivity — leave-one-lineage-out (LOLO).

See also: src/phylo_corrected_convergence.py -- a separate, independent check
(Reviewer 2, point 3) that reuses this script's lineage partition as strata
for a classical Mantel-Haenszel association analysis, but never touches the
VAE. This script is the VAE-based robustness check (Reviewer 1); that one is
not. Neither duplicates the other's calculation.

Phase 1  Define lineage groups from the pre-computed genomic distance matrix
         (sourmash 1 - ANI, 150 E. coli strains).  Average-linkage (UPGMA),
         cut into 5-8 clusters (closest to 6), merge clusters with < 5 strains
         into their nearest neighbour.

         Also builds a Neighbor-Joining tree (Saitou-Nei, self-contained
         numpy implementation -- no new alignment, no IQ-TREE) from the
         SAME distance matrix used for the UPGMA cut above, purely as a
         reviewer-requested cross-check of whether the resulting 5-lineage
         partition still looks reasonable against a continuous-divergence
         topology. This is diagnostic only: the UPGMA assignment computed
         above (`final_map` / 06_lineage_clusters.csv) remains the single
         source of truth that every downstream script (Phase 2/3 below,
         effector_lineage_distribution.py, edge_contingency_lineage_check.py)
         reuses -- the NJ tree does not replace or recompute it.

Phase 2  For each lineage group, rebuild the network on the effector matrix
         WITHOUT that group's strains -- using a SINGLE-SHOT Bernoulli-VAE fit
         (bernoulli_vae_network), the SAME method build_final_network.py uses
         for the reported network, NOT a bootstrap-mean consensus -- and
         compare it to the full 150-strain reference (Pearson on the upper
         triangle, Jaccard of the top-50 edges by |weight|, recall of the
         top-50). Both the reference and the per-lineage reconstructions use
         the single-shot method so this sensitivity analysis is consistent
         with the manuscript's reported network, not the separate
         Bootstrap-Mean consensus method; see consensus() below.

Phase 3  Concrete reviewer cases: lineage membership of E2348/69 (FM180568) and
         Sakai (BA000007); carriage of EspM/EspM1 and NleG/NleD by lineage;
         effect of lineage removal on the EspM-EspM1 and NleG-NleD edges.

Inputs  (relative to the repository root):
    data/genomes/ecoli_150_distance_matrix.csv
    data/genomes/ecoli_150_nj_tree.nwk
    data/input/EffectorRepertoire_Ecoli.xlsx
    results/02_network_final_top30.json   (reference: the reported single-shot
                                         network, from build_final_network.py)
Outputs (results/phylogenetic_sensitivity/):
    06_lineage_clusters.csv, 06_lineage_summary.json (now includes nj_tree_diagnostics)
    06_leave_one_lineage_out.csv          <- the main table
    06_findings_reviewer_cases.json
    06_distance_matrix_provenance_copy.csv, 06_nj_tree_input_provenance_copy.nwk
                                        <- provenance copies of the inputs used
    06_reference_consensus_single_shot.npy, 06_consensus_excl_cluster{c}.npy
    06_nj_tree_150strains.nwk           <- NJ tree built here from
                                           06_distance_matrix_provenance_copy.csv
    TableS8.csv                        <- per-lineage MRCA-purity check on that tree
                                           (manuscript's Table S8; this IS the numbered
                                           supplementary table, kept unrenamed)
Outputs (results/figures/):
    FigureS1.{pdf,png}                 <- full 150-tip NJ phylogram, colored by lineage
                                           (manuscript's Figure S1; this IS the numbered
                                           supplementary figure, kept unrenamed)

Usage:
    python src/phylogenetic_sensitivity.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

import os as _os
_os.environ.setdefault("MPLBACKEND", "Agg")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

torch.set_num_threads(1)  # deterministic + faster on this small VAE

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from inference_functions import load_binary_matrix, bernoulli_vae_network  # noqa: E402
from vae_config_loader import load_vae_params  # noqa: E402
from figure_style import *  # noqa: E402,F401,F403

init_style()

# ── configuration ────────────────────────────────────────────────────────────
DIST_CSV = ROOT / "data" / "genomes" / "ecoli_150_distance_matrix.csv"
TREE_NWK = ROOT / "data" / "genomes" / "ecoli_150_nj_tree.nwk"
INPUT_XLSX = ROOT / "data" / "input" / "EffectorRepertoire_Ecoli.xlsx"
# The leave-one-lineage-out reference must be the network REPORTED in the
# manuscript, 02_network_final_top30.json (single-shot VAE on all 150
# strains, build_final_network.py) -- not
# 01_network_bootstrap_mean_top30_table1_input.json, the separate
# Bootstrap-Mean network that only feeds the Table 1 stability comparison.
REF_JSON = ROOT / "results" / "02_network_final_top30.json"
OUT_DIR = ROOT / "results" / "phylogenetic_sensitivity"
FIGURES_DIR = ROOT / "results" / "figures"

LINEAGE_COLORS = {1: "#4A96D8", 2: "#C85A9A", 3: "#2A9E80", 4: "#D4A080", 5: "#9A90D8"}
LINEAGE_OTHER_COLOR = "#B8B8B8"

MIN_CLUSTER = 5
TARGET_K = 6
K_LOW, K_HIGH = 5, 8

# Loaded from the single shared src/vae_config.json so this script's
# reconstruction method can never silently drift from
# build_final_network.py's (see `consensus()` below).
VAE_PARAMS = load_vae_params()
SEED = 42
TOPK = 50

CASES = {"E2348/69": "FM180568", "Sakai": "BA000007"}
EDGES_OF_INTEREST = [("EspM", "EspM1"), ("NleG", "NleD")]
# ─────────────────────────────────────────────────────────────────────────────


def strip_version(acc):
    return str(acc).split(".")[0]


# ── Neighbor-Joining tree (reviewer resubmission, Part 1) ──────────────────
def build_nj_tree(D, labels):
    """
    Classic Saitou-Nei neighbor-joining, self-contained in numpy so it adds
    no new dependency: built directly from the SAME symmetric genomic
    distance matrix `D` used above for the UPGMA lineage cut (no new
    alignment, no IQ-TREE).

    Returns
    -------
    newick       : str -- Newick string with branch lengths
    depth        : dict[node_id -> float]  cumulative branch length from root
    children     : dict[node_id -> (left, left_len, right, right_len)]
    leaves_under : dict[node_id -> frozenset[str]]  leaf labels below a node
    root_id      : int
    """
    n = len(labels)
    active = list(range(n))
    Dd = {i: {j: float(D[i, j]) for j in range(n) if j != i} for i in range(n)}
    children = {}
    next_id = n

    while len(active) > 2:
        r = len(active)
        total = {i: sum(Dd[i][j] for j in active if j != i) for i in active}
        best = None
        for a in range(len(active)):
            i = active[a]
            for b in range(a + 1, len(active)):
                j = active[b]
                q = (r - 2) * Dd[i][j] - total[i] - total[j]
                if best is None or q < best[0]:
                    best = (q, i, j)
        _, i, j = best

        li = 0.5 * Dd[i][j] + (total[i] - total[j]) / (2 * (r - 2))
        lj = Dd[i][j] - li
        li, lj = max(li, 0.0), max(lj, 0.0)  # NJ can propose slightly negative
                                              # branch lengths from noisy
                                              # distances; clipping to 0 is
                                              # standard practice

        new = next_id
        next_id += 1
        children[new] = (i, li, j, lj)

        Dd[new] = {}
        for k in active:
            if k in (i, j):
                continue
            d_new_k = 0.5 * (Dd[i][k] + Dd[j][k] - Dd[i][j])
            Dd[new][k] = d_new_k
            Dd[k][new] = d_new_k

        del Dd[i], Dd[j]
        active = [x for x in active if x not in (i, j)] + [new]

    i, j = active
    d_ij = Dd[i][j]
    root_id = next_id
    # Arbitrary midpoint rooting of the final unrooted NJ edge (standard
    # convention) -- does not affect topology or total branch length.
    children[root_id] = (i, d_ij / 2.0, j, d_ij / 2.0)

    leaves_under = {}

    def populate(node):
        if node not in children:
            leaves_under[node] = frozenset({labels[node]})
            return leaves_under[node]
        l, _, r, _ = children[node]
        s = populate(l) | populate(r)
        leaves_under[node] = s
        return s

    populate(root_id)

    depth = {}

    def set_depth(node, d):
        depth[node] = d
        if node in children:
            l, ll, r, rl = children[node]
            set_depth(l, d + ll)
            set_depth(r, d + rl)

    set_depth(root_id, 0.0)

    def to_newick(node):
        if node not in children:
            return labels[node]
        l, ll, r, rl = children[node]
        return f"({to_newick(l)}:{ll:.8f},{to_newick(r)}:{rl:.8f})"

    newick = to_newick(root_id) + ";"

    return newick, depth, children, leaves_under, root_id


def nj_tree_analysis(A, labels, final_map, final_clusters):
    """
    Reviewer-resubmission Part 1: build the NJ tree from the SAME 150x150
    sourmash distance matrix already loaded for the UPGMA lineage cut above,
    save it in Newick, draw the full 150-strain tree colored by the CURRENT
    5 UPGMA lineages, report divergence statistics, and give a data-driven
    verdict on whether the 5-lineage cut still looks reasonable against the
    NJ topology.

    Read-only diagnostic layered on top of the UPGMA assignment computed in
    phase1() above (`final_map`) -- does not change it, and does not touch
    build_final_network.py, hierarchical_clustering.py, or any of the
    already-published leave-one-lineage-out numbers.
    """
    print("\n" + "=" * 64)
    print("  NEIGHBOR-JOINING TREE (150 strains, supplementary)")
    print("=" * 64)

    newick, depth, children, leaves_under, root_id = build_nj_tree(A, labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nwk_path = OUT_DIR / "06_nj_tree_150strains.nwk"
    nwk_path.write_text(newick, encoding="utf-8")
    print(f"  -> {nwk_path.relative_to(ROOT)}")

    # ── divergence summary, from the same raw distances both the UPGMA cut
    # and this NJ tree are built from ──────────────────────────────────────
    iu = np.triu_indices(len(labels), k=1)
    pairwise = A[iu]
    mean_dist, max_dist = float(pairwise.mean()), float(pairwise.max())
    tree_depth = max(depth[i] for i in range(len(labels)))  # deepest tip
    print(f"  mean pairwise genomic distance : {mean_dist:.6f}")
    print(f"  max  pairwise genomic distance : {max_dist:.6f}")
    print(f"  total tree depth (root -> deepest tip) : {tree_depth:.6f}")

    # ── lineage coherence: for each of the 5 UPGMA lineages, find its MRCA
    # subtree on the NJ tree and check whether it is PURE (monophyletic --
    # exactly that lineage's strains, no others) or contaminated by outside
    # strains, which is how an artificial cut through a continuum would show
    # up ─────────────────────────────────────────────────────────────────
    def mrca_leaf_set(target_labels):
        target = frozenset(target_labels)
        best = None
        for leafset in leaves_under.values():
            if target <= leafset and (best is None or len(leafset) < len(best)):
                best = leafset
        return best

    print("\n  lineage coherence on the NJ tree (MRCA purity):")
    coherence_rows = []
    for cid, members in final_clusters.items():
        mrca_set = mrca_leaf_set(members)
        extra = len(mrca_set) - len(members)
        pure = extra == 0
        print(f"    lineage {cid} (n={len(members)}): MRCA subtree has "
              f"{len(mrca_set)} tips "
              f"({'monophyletic' if pure else f'+{extra} foreign tips'})")
        coherence_rows.append({
            "lineage": cid, "n_members": len(members),
            "mrca_size": len(mrca_set), "n_foreign_in_mrca": extra,
            "monophyletic": pure,
        })
    coherence_df = pd.DataFrame(coherence_rows)
    # This IS the manuscript's numbered Table S8 -- protected filename, do not rename.
    coherence_df.to_csv(OUT_DIR / "TableS8.csv", index=False)
    print(f"  -> {(OUT_DIR / 'TableS8.csv').relative_to(ROOT)}")

    # ── figure: full 150-tip phylogram, colored by UPGMA lineage ──────────
    leaf_order = []

    def dfs(node):
        if node in children:
            l, _, r, _ = children[node]
            dfs(l)
            dfs(r)
        else:
            leaf_order.append(node)

    dfs(root_id)

    y = {leaf: pos for pos, leaf in enumerate(leaf_order)}

    def set_y(node):
        if node in children:
            l, _, r, _ = children[node]
            set_y(l)
            set_y(r)
            y[node] = (y[l] + y[r]) / 2.0

    set_y(root_id)

    def branch_color(node):
        lids = {final_map[lbl] for lbl in leaves_under[node]}
        return LINEAGE_COLORS.get(next(iter(lids))) if len(lids) == 1 else LINEAGE_OTHER_COLOR

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize(W_FULL, 220))
    for node, (l, ll, r, rl) in children.items():
        ax.plot([depth[node], depth[l]], [y[l], y[l]], color=branch_color(l), linewidth=0.5)
        ax.plot([depth[node], depth[r]], [y[r], y[r]], color=branch_color(r), linewidth=0.5)
        ax.plot([depth[node], depth[node]], [y[l], y[r]], color=branch_color(node), linewidth=0.5)

    ax.set_yticks([y[leaf] for leaf in leaf_order])
    ax.set_yticklabels([labels[leaf] for leaf in leaf_order], fontsize=3)
    ax.set_ylim(-1, len(leaf_order))
    for lid in sorted(final_clusters):
        ax.plot([], [], color=LINEAGE_COLORS.get(lid, LINEAGE_OTHER_COLOR), linewidth=1.5,
                 label=f"Lineage {lid} (n={len(final_clusters[lid])})")
    ax.legend(loc="upper right", fontsize=5)
    style(ax, xlabel="Genomic distance (sourmash, 1 - ANI)", ylabel="Strain")
    # This IS the manuscript's numbered Figure S1 -- protected filename, do not rename.
    save(fig, str(FIGURES_DIR / "FigureS1"), fmt="pdf", dpi=600)
    save(fig, str(FIGURES_DIR / "FigureS1"), fmt="png", dpi=600)
    plt.close(fig)
    print(f"  -> {FIGURES_DIR.relative_to(ROOT)}/FigureS1.{{pdf,png}}")

    n_pure = int(sum(r["monophyletic"] for r in coherence_rows))
    print(f"\n  VERDICT: {n_pure}/5 lineages are exactly monophyletic on the NJ tree "
          f"(mean pairwise dist {mean_dist:.4f}, max {max_dist:.4f}, "
          f"tree depth {tree_depth:.4f}).")

    return {
        "mean_pairwise_distance": mean_dist,
        "max_pairwise_distance": max_dist,
        "tree_depth": tree_depth,
        "n_lineages_monophyletic": n_pure,
        "lineage_coherence": coherence_rows,
        "newick_file": nwk_path.relative_to(ROOT).as_posix(),
        "figure_file": "results/figures/FigureS1.pdf",
    }


# ── Phase 1 ────────────────────────────────────────────────────────────────
def phase1():
    D = pd.read_csv(DIST_CSV, index_col=0, float_precision="round_trip")
    labels_full = list(D.index)
    labels = [strip_version(x) for x in labels_full]
    A = D.to_numpy(dtype=float)
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A[A < 0] = 0.0
    condensed = squareform(A, checks=False)

    Z = linkage(condensed, method="average")
    n = len(labels)
    heights = Z[:, 2]

    candidates = []
    for k in range(2, 16):
        cl = fcluster(Z, t=k, criterion="maxclust")
        kk = int(len(np.unique(cl)))
        if kk != k:
            continue
        cut = float((heights[n - 1 - k] + heights[n - k]) / 2.0)
        sizes = np.bincount(cl)[1:]
        candidates.append((k, cut, cl, kk, int(sizes.min()), sizes))

    in_range = [c for c in candidates if K_LOW <= c[3] <= K_HIGH]
    if in_range:
        in_range.sort(key=lambda c: (abs(c[3] - TARGET_K), -c[4]))
        chosen = in_range[0]
        note = f"cut within range [{K_LOW},{K_HIGH}]"
    else:
        candidates.sort(key=lambda c: (abs(c[3] - TARGET_K), -c[4]))
        chosen = candidates[0]
        note = f"no cut within [{K_LOW},{K_HIGH}]; nearest k to {TARGET_K}"

    k, cut, cl, kk, minsize, sizes = chosen
    print(f"[phase1] {n} strains | average linkage | {note}")
    print(f"[phase1] cut height = {cut:.6f} -> {kk} clusters, sizes={sorted(sizes, reverse=True)}")

    clusters = {}
    for lab, c in zip(labels, cl):
        clusters.setdefault(int(c), []).append(lab)

    merges = []

    def cluster_centroid_dist(ca, cb):
        ia = [labels.index(x) for x in ca]
        ib = [labels.index(x) for x in cb]
        return float(A[np.ix_(ia, ib)].mean())

    while True:
        small = [c for c, m in clusters.items() if len(m) < MIN_CLUSTER]
        if not small:
            break
        c_small = min(small, key=lambda c: len(clusters[c]))
        others = [c for c in clusters if c != c_small]
        nearest = min(others, key=lambda c: cluster_centroid_dist(clusters[c_small], clusters[c]))
        d = cluster_centroid_dist(clusters[c_small], clusters[nearest])
        merges.append({
            "merged_cluster": c_small,
            "n_strains_merged": len(clusters[c_small]),
            "into_cluster": nearest,
            "avg_linkage_distance": round(d, 6),
            "strains_merged": sorted(clusters[c_small]),
        })
        print(f"[phase1] merge cluster {c_small} (n={len(clusters[c_small])}) -> {nearest} (dist {d:.5f})")
        clusters[nearest].extend(clusters[c_small])
        del clusters[c_small]

    ordered = sorted(clusters.values(), key=len, reverse=True)
    final_map, final_clusters = {}, {}
    for new_id, members in enumerate(ordered, start=1):
        final_clusters[new_id] = sorted(members)
        for m in members:
            final_map[m] = new_id

    print("\n=== FINAL LINEAGE CLUSTERS ===")
    for cid, members in final_clusters.items():
        print(f"  Cluster {cid}: n={len(members)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"strain": s, "accession_full": lf, "lineage_cluster": final_map[strip_version(lf)]}
            for s, lf in zip(labels, labels_full)]
    pd.DataFrame(rows).sort_values(["lineage_cluster", "strain"]).to_csv(
        OUT_DIR / "06_lineage_clusters.csv", index=False)

    nj_diagnostics = nj_tree_analysis(A, labels, final_map, final_clusters)

    summary = {
        "distance_matrix": "data/genomes/ecoli_150_distance_matrix.csv",
        "linkage_method": "average",
        "k_range_target": [K_LOW, K_HIGH],
        "k_preferred": TARGET_K,
        "cut_height": cut,
        "n_clusters_before_merge": int(kk),
        "note": note,
        "min_cluster_size_rule": MIN_CLUSTER,
        "merges": merges,
        "n_clusters_final": len(final_clusters),
        "cluster_sizes_final": {str(c): len(m) for c, m in final_clusters.items()},
        "clusters_final": {str(c): m for c, m in final_clusters.items()},
        "nj_tree_diagnostics": nj_diagnostics,
    }
    with open(OUT_DIR / "06_lineage_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # provenance copies of the exact inputs consumed
    shutil.copyfile(DIST_CSV, OUT_DIR / "06_distance_matrix_provenance_copy.csv")
    shutil.copyfile(TREE_NWK, OUT_DIR / "06_nj_tree_input_provenance_copy.nwk")
    print(f"-> {OUT_DIR.relative_to(ROOT)}/06_lineage_clusters.csv, 06_lineage_summary.json, "
          f"06_distance_matrix_provenance_copy.csv, 06_nj_tree_input_provenance_copy.nwk")


# ── Phase 2 + 3 helpers ───────────────────────────────────────────────────
def upper_tri(arr):
    ii, jj = np.triu_indices(arr.shape[0], k=1)
    return arr[ii, jj]


def top_edge_set(arr, k=TOPK):
    ii, jj = np.triu_indices(arr.shape[0], k=1)
    values = np.abs(arr[ii, jj])
    top_idx = np.argsort(values)[-k:]
    return {tuple(sorted((int(ii[idx]), int(jj[idx])))) for idx in top_idx}


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else np.nan


def own_top30_edge_set(arr, effectors, top_percent=30):
    """
    Threshold `arr` (63x63 signed similarity matrix) at ITS OWN top-30% cut,
    identical procedure to build_final_network.py / export_grafo_json_topk:
    rank |weight| over the upper-triangle, EXCLUDE exact zeros before
    cutting, keep the top `top_percent`% BY COUNT among the non-zero values,
    threshold = the k-th largest observed value (not np.percentile), retain
    with >= threshold (inclusive). Returns a set of (effector_a, effector_b)
    sorted-name tuples -- the edge set as it would be exported for THIS
    matrix alone, on its own scale, not compared against any fixed
    reference threshold.
    """
    ii, jj = np.triu_indices(arr.shape[0], k=1)
    abs_tri = np.abs(arr[ii, jj])
    nonzero = abs_tri[abs_tri > 0]
    k = max(int(len(nonzero) * top_percent / 100), 1)
    threshold = np.sort(nonzero)[-k]
    return {
        tuple(sorted((effectors[ii[t]], effectors[jj[t]])))
        for t in np.where(abs_tri >= threshold)[0]
    }


def consensus(matrix, tag, cache: Path):
    """
    Build the 63x63 signed network for `matrix` (all strains, or all strains
    minus one excluded lineage).

    Both the reference and every lineage-excluded network must be built
    EXACTLY the way build_final_network.py builds the reported network
    (02_network_final_top30.json, REF_JSON above): one Bernoulli-VAE fit
    (bernoulli_vae_network) on all strains present in `matrix`, raw
    (unnormalized) signed cosine similarity (normalize_adj=False), no
    bootstrap averaging -- otherwise this sensitivity analysis would
    silently test a different method (a Bootstrap-Mean consensus) than the
    single-shot network actually reported. `tag` names the cache files
    (06_reference_consensus_single_shot.npy, 06_consensus_excl_cluster{c}.npy);
    if this function's method ever changes, stale cached .npy files under
    those names must be deleted so they get recomputed.
    """
    if cache.is_file():
        print(f"  [cache] {cache.name}")
        return np.load(cache)
    t = time.time()
    adj_df, _, _ = bernoulli_vae_network(
        matrix, **VAE_PARAMS, device="cpu", seed=SEED, verbose=False,
        normalize_adj=False,
    )
    arr = adj_df.values.astype(float)
    np.save(cache, arr)
    print(f"  {tag}: {matrix.shape[1]} strains -> 63x63 single-shot VAE in {(time.time()-t)/60:.1f} min")
    return arr


def edge_index(effectors, a, b):
    return effectors.index(a), effectors.index(b)


# ── Phase 2 + 3 ───────────────────────────────────────────────────────────
def phase2_phase3():
    matrix = load_binary_matrix(INPUT_XLSX)
    effectors = matrix.index.tolist()
    print(f"[phase2] matrix {matrix.shape[0]} effectors x {matrix.shape[1]} strains")

    clusters_df = pd.read_csv(OUT_DIR / "06_lineage_clusters.csv")
    clusters = {int(c): sorted(g["strain"]) for c, g in clusters_df.groupby("lineage_cluster")}
    print(f"[phase2] {len(clusters)} lineage clusters: " +
          ", ".join(f"C{c}(n={len(m)})" for c, m in clusters.items()))

    ref = consensus(matrix, "REF", OUT_DIR / "06_reference_consensus_single_shot.npy")

    json_pairs = set()
    overlap = float("nan")
    if REF_JSON.is_file():
        ref_edges_json = json.load(open(REF_JSON, encoding="utf-8"))["elements"]["edges"]
        json_pairs = {tuple(sorted((e["data"]["source"], e["data"]["target"])))
                      for e in ref_edges_json}
        ii, jj = np.triu_indices(ref.shape[0], k=1)
        abs_tri = np.abs(ref[ii, jj])
        k30 = max(int(len(abs_tri[abs_tri > 0]) * 30 / 100), 1)
        thr30 = np.sort(abs_tri[abs_tri > 0])[-k30]
        my_top30 = {tuple(sorted((effectors[ii[t]], effectors[jj[t]])))
                    for t in np.where(abs_tri >= thr30)[0]}
        overlap = len(my_top30 & json_pairs) / len(json_pairs)
        print(f"[phase2] top-30% reproduced: {len(my_top30)} edges, "
              f"overlap with reference JSON ({len(json_pairs)}): {overlap:.3%}")
    else:
        print(f"[phase2] reference JSON not found ({REF_JSON.name}); skipping top-30 overlap check")

    ii, jj = np.triu_indices(ref.shape[0], k=1)
    ref_top50 = top_edge_set(ref, TOPK)
    ref_tri = upper_tri(ref)
    if json_pairs:
        json_mask = np.array([tuple(sorted((effectors[ii[t]], effectors[jj[t]]))) in json_pairs
                              for t in range(len(ii))])
    else:
        json_mask = np.ones(len(ii), dtype=bool)

    ref_edge_w = {f"{a}-{b}": float(ref[edge_index(effectors, a, b)])
                  for a, b in EDGES_OF_INTEREST}
    order = np.argsort(-np.abs(ref_tri))
    rank_of = {(min(ii[o], jj[o]), max(ii[o], jj[o])): r for r, o in enumerate(order, 1)}
    ref_edge_rank = {}
    for a, b in EDGES_OF_INTEREST:
        x, y = edge_index(effectors, a, b)
        ref_edge_rank[f"{a}-{b}"] = int(rank_of[(min(x, y), max(x, y))])

    rows, per_cluster_edges = [], {}
    for c, strains in clusters.items():
        excl = [s for s in strains if s in matrix.columns]
        sub = matrix.drop(columns=excl)
        print(f"\n[C{c}] drop {len(excl)} strains -> {sub.shape[1]} strains remain")
        cons_c = consensus(sub, f"C{c}", OUT_DIR / f"06_consensus_excl_cluster{c}.npy")

        pear_full = float(pearsonr(ref_tri, upper_tri(cons_c))[0])
        pear_refedges = float(pearsonr(ref_tri[json_mask], upper_tri(cons_c)[json_mask])[0])
        top50_c = top_edge_set(cons_c, TOPK)
        jac50 = float(jaccard(ref_top50, top50_c))
        recall50 = len(ref_top50 & top50_c) / len(ref_top50)

        # NEW metric: re-threshold this lineage-excluded network at ITS OWN
        # top-30% cut (same count-based, non-zero-excluding procedure as
        # build_final_network.py, applied fresh to the data without this
        # lineage -- NOT the reference's fixed threshold applied to this
        # matrix), then Jaccard against the 585 edges of the ACTUAL reported
        # network (02_network_final_top30.json / `json_pairs`). This tests
        # whether the reported edge SET is robust to lineage removal even
        # when each exclusion gets to pick its own natural cutoff, as
        # opposed to jaccard_top50_vs_red_completa above (top-50 by rank,
        # both networks on the reference's own absolute-weight scale).
        own_top30_c = own_top30_edge_set(cons_c, effectors, top_percent=30)
        jac_final_top30 = float(jaccard(json_pairs, own_top30_c)) if json_pairs else np.nan

        rows.append({
            "cluster_excluido": c,
            "n_cepas_excluidas": len(excl),
            "n_cepas_restantes": sub.shape[1],
            "pearson_vs_red_completa": round(pear_full, 4),
            "jaccard_top50_vs_red_completa": round(jac50, 4),
            "pearson_solo_aristas_ref(JSON)": round(pear_refedges, 4),
            "recall_top50": round(recall50, 4),
            "jaccard_red_final_top30": round(jac_final_top30, 4),
            "n_aristas_propio_top30": len(own_top30_c),
        })

        edinfo = {}
        for a, b in EDGES_OF_INTEREST:
            x, y = edge_index(effectors, a, b)
            w_ref, w_c = float(ref[x, y]), float(cons_c[x, y])
            edinfo[f"{a}-{b}"] = {
                "w_ref": round(w_ref, 4), "w_excl": round(w_c, 4),
                "delta": round(w_c - w_ref, 4),
                "rel_change_%": round(100 * (w_c - w_ref) / (abs(w_ref) + 1e-9), 1),
                "in_ref_top50": (min(x, y), max(x, y)) in ref_top50,
                "in_excl_top50": (min(x, y), max(x, y)) in top50_c,
            }
        per_cluster_edges[c] = edinfo
        print(f"      Pearson(full)={pear_full:.4f}  Jaccard top50={jac50:.4f}  recall={recall50:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "06_leave_one_lineage_out.csv", index=False)
    print("\n==== PHASE 2 TABLE ====")
    print(table.to_string(index=False))

    clu_of = dict(zip(clusters_df["strain"], clusters_df["lineage_cluster"]))
    espm_carriers = {}
    for eff in ["EspM", "EspM1"]:
        carriers = matrix.columns[matrix.loc[eff] == 1].tolist()
        by_clu = pd.Series([clu_of[s] for s in carriers]).value_counts().sort_index().to_dict()
        espm_carriers[eff] = {"n": len(carriers), "por_cluster": {int(k): int(v) for k, v in by_clu.items()}}
    nle_carriers = {}
    for eff in ["NleG", "NleD"]:
        carriers = matrix.columns[matrix.loc[eff] == 1].tolist()
        by_clu = pd.Series([clu_of[s] for s in carriers]).value_counts().sort_index().to_dict()
        nle_carriers[eff] = {"n": len(carriers), "por_cluster": {int(k): int(v) for k, v in by_clu.items()}}

    findings = {
        "config": {
            "vae_params": VAE_PARAMS, "seed": SEED,
            "method": "bernoulli_vae_network single-shot (normalize_adj=False), "
                      "same as build_final_network.py's reported network",
            "topk": TOPK,
        },
        "referencia": {
            "fuente": "bernoulli_vae_network single-shot (150 strains, seed 42) "
                     "-- matches build_final_network.py's reported network",
            "solapamiento_top30_con_JSON": round(overlap, 4) if overlap == overlap else None,
            "n_aristas_JSON": len(json_pairs),
            "aristas_interes_en_referencia": {
                k: {"peso": ref_edge_w[k], "rank_por_|peso|": ref_edge_rank[k],
                    "en_top50": ref_edge_rank[k] <= 50}
                for k in ref_edge_w
            },
        },
        "fase2_tabla": rows,
        "fase2_aristas_interes_por_cluster": per_cluster_edges,
        "fase3": {
            "casos_revisor1": {
                nombre: {"accession": acc, "lineage_cluster": int(clu_of.get(acc, -1))}
                for nombre, acc in CASES.items()
            },
            "portadores_espM": espm_carriers,
            "portadores_NleG_NleD": nle_carriers,
        },
    }
    with open(OUT_DIR / "06_findings_reviewer_cases.json", "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2, ensure_ascii=False)

    print("\n==== PHASE 3 ====")
    for nombre, acc in CASES.items():
        print(f"  {nombre} ({acc}) -> lineage cluster {clu_of.get(acc)}")
    print(f"-> {OUT_DIR.relative_to(ROOT)}/06_leave_one_lineage_out.csv, 06_findings_reviewer_cases.json")


def main() -> int:
    print("=" * 64)
    print("  PHYLOGENETIC SENSITIVITY  (leave-one-lineage-out)")
    print("=" * 64)
    phase1()
    phase2_phase3()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
