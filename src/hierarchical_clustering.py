"""
Hierarchical clustering of the signed effector network (k = 7 modules).

Works exclusively from the exported network JSON: rebuilds the 63x63 signed
similarity matrix from the edges, turns it into a signed cosine distance
(d = 1 - s), runs average-linkage hierarchical clustering, sweeps the
silhouette score for k = 2..15 and cuts at the fixed k = 7.

k = 7 is DEFINITIVE, re-derived from the CURRENT 02_network_final_top30.json
(585 edges, |weight| threshold 0.286461). With this network, the silhouette
sweep's GLOBAL maximum is at k = 7 (~0.260), not a fragmentation-inflated
large k: cluster sizes at k=7 are [23, 19, 9, 4, 3, 3, 2] -- only one small
(<=2-effector) cluster. Compare to k=14 (silhouette ~0.250, LOWER than k=7): 9 of its 14
clusters have only 1-2 effectors -- that is the fragmentation-inflation
failure mode this analysis explicitly avoids, and k=14 is not even the
global maximum with this network. k=8 (silhouette ~0.257, also below k=7)
introduces a first singleton cluster. k=7 is therefore both the best-scoring
and the least-fragmented reasonable choice.

Inputs  (relative to the repository root):
    results/02_network_final_top30.json
Outputs:
    results/04_clustering_silhouette_scores.csv
    results/TableS9.csv                              (manuscript's Table S9)
    results/04_clustering_structural_balance.csv
    results/04_network_final_clustered.json
    results/figures/FigureS2a.{pdf,png}               (manuscript's Figure S2a:
                                                        dendrogram)
    results/figures/FigureS2b.{pdf,png}               (manuscript's Figure S2b:
                                                        silhouette vs k)

Usage:
    python src/hierarchical_clustering.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from figure_style import *  # noqa: E402,F401,F403

init_style()

# ── configuration ────────────────────────────────────────────────────────────
JSON_PATH = ROOT / "results" / "02_network_final_top30.json"
OUTPUT_JSON = ROOT / "results" / "04_network_final_clustered.json"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "results" / "figures"

N_CLUSTERS = 7            # DEFINITIVE (was 6; see module docstring CHANGE note)
LINKAGE_METHOD = "average"
DENDROGRAM_PALETTE = [
    "#E8A0C8", "#C85A9A", "#90C4F4", "#4A96D8", "#9A90D8",
    "#5A4AAE", "#6ACFB0", "#2A9E80", "#D4A080", "#B06A40",
]
DENDROGRAM_ABOVE_THRESHOLD_COLOR = "#B8B8B8"
CLUSTER_COLOR_MAP = {
    5: DENDROGRAM_PALETTE[0], 1: DENDROGRAM_PALETTE[1], 8: DENDROGRAM_PALETTE[2],
    2: DENDROGRAM_PALETTE[3], 3: DENDROGRAM_PALETTE[4], 4: DENDROGRAM_PALETTE[5],
    6: DENDROGRAM_PALETTE[6], 7: DENDROGRAM_PALETTE[7], 9: DENDROGRAM_PALETTE[8],
}
# ─────────────────────────────────────────────────────────────────────────────

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. load network ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  LOADING NETWORK FROM JSON")
print("=" * 60)

with open(JSON_PATH, encoding="utf-8") as f:
    network = json.load(f)

nodes = [n["data"]["id"] for n in network["elements"]["nodes"]]
edges = network["elements"]["edges"]
n = len(nodes)
idx = {name: i for i, name in enumerate(nodes)}
print(f"  nodes : {n}")
print(f"  edges : {len(edges)}")

# ── 2. rebuild similarity matrix from edges ─────────────────────────────────
sim = np.zeros((n, n))
np.fill_diagonal(sim, 1.0)
n_pos = n_neg = 0
for edge in edges:
    i, j = idx[edge["data"]["source"]], idx[edge["data"]["target"]]
    w = float(edge["data"]["weight"])
    sim[i, j] = sim[j, i] = w
    n_pos += w >= 0
    n_neg += w < 0
print(f"\n  positive edges : {n_pos}")
print(f"  negative edges : {n_neg}")

# ── 3. signed cosine distance ──────────────────────────────────────────────
dist = 1.0 - sim
dist = (dist + dist.T) / 2
np.fill_diagonal(dist, 0.0)
dist = np.clip(dist, 0, None)
dist_condensed = squareform(dist, checks=False)

# ── 4. hierarchical clustering + silhouette sweep ──────────────────────────
Z = linkage(dist_condensed, method=LINKAGE_METHOD)

silhouette_scores = {}
max_k = min(15, n - 1)
for k in range(2, max_k + 1):
    k_labels = fcluster(Z, k, criterion="maxclust")
    if len(np.unique(k_labels)) < 2:
        continue
    silhouette_scores[k] = silhouette_score(dist, k_labels, metric="precomputed")

print("\n" + "=" * 60)
print("  SILHOUETTE SCORE BY NUMBER OF CLUSTERS")
print("=" * 60)
for k, score in silhouette_scores.items():
    print(f"  k={k:>2}: silhouette={score:.4f}")

fig, ax = plt.subplots(figsize=figsize(W_70, H_STD))
ax.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()),
        marker="o", color="#4A96D8", linewidth=1.2)
ax.axvline(N_CLUSTERS, color="#C85A9A", linestyle="--", linewidth=1.0)
ax.set_xticks(list(silhouette_scores.keys()))
style(ax, xlabel="Number of clusters", ylabel="Silhouette score")
base_path = str(FIGURES_DIR / "FigureS2b")
save(fig, base_path, fmt="pdf", dpi=600)
save(fig, base_path, fmt="png", dpi=600)
plt.close(fig)

labels = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")
cluster_labels = {eff: int(lab) for eff, lab in zip(nodes, labels)}

link_members = {}
for i, row in enumerate(Z):
    left, right = int(row[0]), int(row[1])
    node_id = n + i
    left_members = {left} if left < n else link_members[left]
    right_members = {right} if right < n else link_members[right]
    link_members[node_id] = left_members | right_members


def cluster_link_color(node_id):
    member_clusters = {int(labels[i]) for i in link_members[node_id]}
    if len(member_clusters) == 1:
        return CLUSTER_COLOR_MAP.get(member_clusters.pop(), DENDROGRAM_ABOVE_THRESHOLD_COLOR)
    return DENDROGRAM_ABOVE_THRESHOLD_COLOR


# ── 5. cluster composition ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  CLUSTER COMPOSITION  ({LINKAGE_METHOD} linkage, {N_CLUSTERS} clusters)")
print("=" * 60)

cluster_df = pd.DataFrame({"effector": nodes, "cluster": labels}).sort_values(
    ["cluster", "effector"]
)
for cid in sorted(cluster_df["cluster"].unique()):
    members = cluster_df[cluster_df["cluster"] == cid]["effector"].tolist()
    print(f"\n  Cluster {cid}  ({len(members)} effectors):")
    for m in members:
        print(f"    {m}")

# ── 6. structural balance test ────────────────────────────────────────────
neg_intra = neg_inter = pos_intra = pos_inter = 0
for edge in edges:
    w = float(edge["data"]["weight"])
    same = cluster_labels[edge["data"]["source"]] == cluster_labels[edge["data"]["target"]]
    if w < 0:
        neg_intra += same
        neg_inter += not same
    else:
        pos_intra += same
        pos_inter += not same

total_neg = neg_intra + neg_inter
total_pos = pos_intra + pos_inter

print("\n" + "=" * 60)
print("  STRUCTURAL BALANCE ANALYSIS")
print("=" * 60)
if total_pos:
    print(f"  positive within  : {pos_intra}/{total_pos} ({100*pos_intra/total_pos:.1f}%)")
    print(f"  positive between : {pos_inter}/{total_pos} ({100*pos_inter/total_pos:.1f}%)")
if total_neg:
    print(f"  negative within  : {neg_intra}/{total_neg} ({100*neg_intra/total_neg:.1f}%)")
    print(f"  negative between : {neg_inter}/{total_neg} ({100*neg_inter/total_neg:.1f}%)")
    balance = neg_inter / total_neg
    print(f"  balance score    : {balance:.3f}")

# ── 6b. export CSVs ──────────────────────────────────────────────────────
pd.DataFrame({
    "k": list(silhouette_scores.keys()),
    "silhouette_score": list(silhouette_scores.values()),
}).to_csv(RESULTS_DIR / "04_clustering_silhouette_scores.csv", index=False)

cluster_df.to_csv(RESULTS_DIR / "TableS9.csv", index=False)

balance_df = pd.DataFrame(
    [
        ("positive_within", pos_intra, total_pos),
        ("positive_between", pos_inter, total_pos),
        ("negative_within", neg_intra, total_neg),
        ("negative_between", neg_inter, total_neg),
    ],
    columns=["edge_type", "count", "total"],
)
balance_df["fraction"] = balance_df["count"] / balance_df["total"]
balance_df.to_csv(RESULTS_DIR / "04_clustering_structural_balance.csv", index=False)
print(f"\n  -> 04_clustering_silhouette_scores.csv / TableS9.csv / 04_clustering_structural_balance.csv")

# ── 7. dendrogram ────────────────────────────────────────────────────────
cross_species_effectors = {
    "SopA Shigella/NleL-2", "IpaH?", "OspG", "OspG/NleH", "NleL-like?", "DUF1778",
}
display_labels = [
    "*SopA/NleL-2" if node == "SopA Shigella/NleL-2"
    else "*IpaH" if node == "IpaH?"
    else "*NleL" if node == "NleL-like?"
    else f"*{node}" if node in cross_species_effectors
    else node
    for node in nodes
]

fig, ax = plt.subplots(figsize=figsize(W_FULL, H_WIDE))
dendrogram(
    Z, labels=display_labels, leaf_rotation=90, leaf_font_size=TICK,
    link_color_func=cluster_link_color,
    above_threshold_color=DENDROGRAM_ABOVE_THRESHOLD_COLOR, ax=ax,
)
style(ax, xlabel="Effector", ylabel="Distance")
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor", fontsize=TICK)
for tick in ax.get_xticklabels():
    if tick.get_text().startswith("*"):
        tick.set_fontweight("bold")
for line in ax.get_lines():
    line.set_linewidth(1.3)
save(fig, str(FIGURES_DIR / "FigureS2a"), fmt="pdf", dpi=600)
save(fig, str(FIGURES_DIR / "FigureS2a"), fmt="png", dpi=600)
plt.close(fig)

# ── 8. export clustered JSON ─────────────────────────────────────────────
for node in network["elements"]["nodes"]:
    node["data"]["cluster"] = cluster_labels[node["data"]["id"]]
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(network, f, indent=4)
print(f"\n  -> {OUTPUT_JSON.relative_to(ROOT)}")
