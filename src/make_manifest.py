#!/usr/bin/env python
"""
make_manifest.py — write MANIFEST.csv: one row per file under results/, with
its sha256, size, generation date (file mtime) and the key figure it carries.

    python src/make_manifest.py

Columns: archivo, ruta_relativa, sha256, bytes, fecha_generacion, cifra_clave
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MANIFEST = os.path.join(ROOT, "MANIFEST.csv")

# Curated key figure per result file (relative path, forward slashes).
CIFRA = {
    "results/02_network_final_signed_similarity.csv":
        "Reported network: 63x63 signed raw cosine similarity (single-shot VAE, 150 strains). 1953 non-zero pairs.",
    "results/02_edge_confidence.csv":
        "1953 pairs: single-shot value + bootstrap sd over 75 replicas (fixed-data bootstrap, seed=42).",
    "results/02_S_boot_reference_stack.npy":
        "Stack (75,63,63) of the 75 bootstrap replicas; source of sd_bootstrap. Do not retrain.",
    "results/02_network_final_top30.json":
        "{n_edges} edges (top-30% by count). |w| threshold {thr}. {pos} positive / {neg} negative. 63 nodes. REPORTED NETWORK.",
    "results/02_threshold_sensitivity.csv":
        "Percentile sweep on the reported network; p70 row -> threshold {thr70}, {e70} edges, 0 isolated.",
    "results/TableS3.csv":
        "{n_neg} negative edges analysed (== negatives of the reported network). {genuine} genuine exclusion / {borderline} borderline / {artefact} artefact.",
    "results/01_method_stability_comparison_table1_input.csv":
        "Table 1: pairwise stability of 7 methods (Pearson, top-50 Jaccard) over 30 outer subsamples (fraction 0.8, seed 42).",
    "results/01_network_bootstrap_mean_top30_table1_input.json":
        "{n_edges} edges (top-30% by |w|), {pos} pos / {neg} neg. Bootstrap-Mean network; feeds Table 1, NOT the reported network.",
    "results/TableS9.csv":
        "63 effectors in {k} modules (average linkage, signed cosine distance).",
    "results/04_clustering_structural_balance.csv":
        "Structural balance k=7: fraction of negative edges falling between modules.",
    "results/04_clustering_silhouette_scores.csv":
        "Silhouette k=2..15; k=7 selected (global maximum, 0.260).",
    "results/04_network_final_clustered.json":
        "02_network_final_top30 edges + 'cluster' node attribute. k=7 modules.",
    "results/ground_truth_validation/05_ground_truth_per_replicate_results.csv":
        "35 replicates x 6 methods: AUC-PR, FPR-indirect, FPR-latent per replicate (seeds 100..134).",
    "results/ground_truth_validation/05_ground_truth_summary.csv":
        "Per-method mean/sd/count of AUC-PR and FPR over 35 replicates.",
    "results/ground_truth_validation/TableS5.csv":
        "Primary criterion: Wilcoxon signed-rank on fpr_latent, VAE vs {vae_pairs}. Raw p + Holm-Bonferroni + significance flag.",
    "results/phylogenetic_sensitivity/06_leave_one_lineage_out.csv":
        "5 lineage groups excluded one at a time: Pearson / top-50 Jaccard / recall of the network vs the full 150-strain reference.",
    "results/phylogenetic_sensitivity/06_distance_matrix_provenance_copy.csv":
        "sourmash (1 - ANI) genomic distance matrix, 150 E. coli strains (provenance copy of the input).",
    "results/phylogenetic_sensitivity/06_nj_tree_input_provenance_copy.nwk":
        "Neighbour-joining tree from the genomic distance matrix (provenance copy of the input).",
    "results/phylogenetic_sensitivity/06_lineage_clusters.csv":
        "strain -> lineage cluster (UPGMA cut into 5-8 groups, clusters < 5 strains merged into nearest).",
    "results/phylogenetic_sensitivity/06_lineage_summary.json":
        "Phase-1 parameters: cut height, cluster sizes, merges.",
    "results/phylogenetic_sensitivity/06_findings_reviewer_cases.json":
        "Phase 2/3 findings: reference top-30 overlap, per-cluster effect on EspM-EspM1 / NleG-NleD, reviewer strain cases.",
    "results/phylo_corrected_convergence/12_ari_permutation_test.csv":
        "Mantel-Haenszel-derived (11-cluster) assignment vs the VAE's 7 modules: adjusted Rand index=0.16, empirical p=0.0004 (10000 relabelings). L1 diagnostic comparator (excluded from main analysis): ARI=0.02, p=0.223 (not significant).",
    "results/figures/FigureS2a.pdf":
        "Supplementary dendrogram: average-linkage clustering of the reported network, coloured by the 7 modules.",
    "results/figures/FigureS2b.pdf":
        "Silhouette score vs number of clusters (k = 2..15), vertical line at k = 7.",
    "results/figures/FigureS2b.png":
        "Silhouette score vs number of clusters (k = 2..15), vertical line at k = 7.",
    # -- pipeline_main.py intermediates (blocks 1-8; backfilled from the Via-B
    #    full run_all.py reproduction, 2026-09-11) --
    "results/01_binary_matrix.csv":
        "63 x 150 binary effector presence/absence matrix (Block 1). Min repertoire 4 (strain LT903847).",
    "results/01_binary_strains_effectors.csv":
        "Same 63 x 150 binary matrix as 01_binary_matrix.csv, effector-name-indexed export of Block 1.",
    "results/01_strains_per_effector.csv":
        "Per-effector strain count (prevalence) out of 150.",
    "results/01_vae_hyperparameter_search_results.csv":
        "48 configs x 10 seeds; best = latent_dim=32, hidden_dim=64, beta_kl=0.5, mean_stability_r=0.9911.",
    "results/01_permutation_control_results_rowwise.csv":
        "Row-wise permutation control: stability real=0.9495 vs permuted=0.4568 (structure destroyed).",
    "results/01_bootstrap_convergence_results.csv":
        "Inner bootstrap convergence (mean/std corr, RMSE) swept over B in {10,25,50,75,100} x fraction in {0.5,0.6,0.8}.",
    "results/01_bootstrap_mean_per_edge_stats.csv":
        "Per-edge mu (mean) and sigma (sd) of the Bootstrap-Mean network across the 30 outer subsamples.",
    "results/01_jaccard_differential_summary.csv":
        "Bootstrap-Mean vs Jaccard top-30%: {bm} vs {jac} edges, {bm_only} BM-only / {jac_only} Jaccard-only.",
    "results/01_jaccard_differential_edges.csv":
        "The {bm_only} Bootstrap-Mean-only and {jac_only} Jaccard-only edges, with both scores and mean prevalence.",
    "results/01_threshold_sensitivity_bootstrap_mean.csv":
        "Percentile sweep (Block 8) on the Bootstrap-Mean network (NOT the reported network); p70 reproduces its top-30% export.",
    "results/01_S_vae_outer.npy":
        "Stack (30,63,63) of the single-shot VAE network across the 30 outer subsamples (Block 5 tail).",
    "results/01_S_bm_outer.npy":
        "Stack (30,63,63) of the Bootstrap-Mean network across the 30 outer subsamples (Block 5 tail).",
    "results/figures/effector_presence_matrix.pdf":
        "Effector presence/absence matrix across the 150 strains (Block 2).",
    "results/figures/effector_similarity_heatmap.png":
        "Heatmap of the Bootstrap-Mean similarity matrix, 150-strain reference (Block 7).",
    "results/figures/std_singleVAE.pdf":
        "Per-edge bootstrap std histogram, single-shot VAE (outer subsamples).",
    "results/figures/std_mean_singleVAE.pdf":
        "Per-edge mean-vs-std scatter, single-shot VAE (outer subsamples), top-50 highlighted.",
    "results/figures/std_bootstrap.pdf":
        "Per-edge bootstrap std histogram, Bootstrap-Mean network (outer subsamples).",
    "results/figures/std_mean_bootstrap.pdf":
        "Per-edge mean-vs-std scatter, Bootstrap-Mean network (outer subsamples), top-50 highlighted.",
    "results/figures/FigureS3a.pdf":
        "Inner bootstrap convergence: mean pairwise correlation vs B, selected B=75/frac=0.6 highlighted.",
    "results/figures/FigureS3b.pdf":
        "Inner bootstrap convergence: RMSE-to-mean vs B, selected B=75/frac=0.6 highlighted.",
    "results/phylogenetic_sensitivity/06_reference_consensus_single_shot.npy":
        "Bootstrap-Mean consensus (63x63) on all 150 strains; LOLO reference matrix.",
    "results/phylogenetic_sensitivity/06_consensus_excl_cluster1.npy":
        "Bootstrap-Mean consensus (63x63) with lineage cluster 1 (75 strains) excluded.",
    "results/phylogenetic_sensitivity/06_consensus_excl_cluster2.npy":
        "Bootstrap-Mean consensus (63x63) with lineage cluster 2 (52 strains) excluded.",
    "results/phylogenetic_sensitivity/06_consensus_excl_cluster3.npy":
        "Bootstrap-Mean consensus (63x63) with lineage cluster 3 (9 strains) excluded.",
    "results/phylogenetic_sensitivity/06_consensus_excl_cluster4.npy":
        "Bootstrap-Mean consensus (63x63) with lineage cluster 4 (7 strains) excluded.",
    "results/phylogenetic_sensitivity/06_consensus_excl_cluster5.npy":
        "Bootstrap-Mean consensus (63x63) with lineage cluster 5 (7 strains) excluded.",
    "results/phylogenetic_sensitivity/pair_lineage_counts.csv":
        "NleG/NleD and EspM/EspM1 per lineage (inside / excluded): lineage 1 inside = N 75, NleG 37, NleD 14, both 0, expected 6.91, P 1.7e-5.",
    "results/phylogenetic_sensitivity/ancestral_reconstruction/events.csv":
        "Fitch-parsimony changes on the midpoint-rooted NJ tree: NleG 3, NleD 11, EspM 8, EspM1 10 (gains+losses).",
    "results/phylogenetic_sensitivity/ancestral_reconstruction/fitch_validation.txt":
        "Own Fitch vs Bio.Phylo ParsimonyScorer: identical minimum changes for all 63 effector families.",
    "results/phylogenetic_sensitivity/ancestral_reconstruction/TableS14.csv":
        "Table S14: carriers and minimum Fitch transitions for the 63 effector families (1 to 27; median 12).",
    "results/phylogenetic_sensitivity/ancestral_reconstruction/TableS14.xlsx":
        "Table S14 as .xlsx, with a 'Notes' sheet holding the legend (same numbers as TableS14.csv).",
    "results/phylogenetic_sensitivity/ancestral_reconstruction/tree_NleG_NleD.png":
        "NJ tree, tips coloured by NleG / NleD presence (0 co-occurrences; NleG 39, NleD 72 strains).",
    "results/phylogenetic_sensitivity/ancestral_reconstruction/tree_EspM_EspM1.png":
        "NJ tree, tips coloured by EspM / EspM1 presence (10 co-occurrences; EspM 28, EspM1 108 strains).",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_edge_stats(path):
    net = json.load(open(path, encoding="utf-8"))
    edges = net["elements"]["edges"]
    pos = sum(1 for e in edges if e["data"]["weight"] >= 0)
    neg = len(edges) - pos
    thr = min(abs(e["data"]["weight"]) for e in edges) if edges else float("nan")
    return dict(n_edges=len(edges), pos=pos, neg=neg, thr=f"{thr:.6f}")


def _fill(rel: str, template: str) -> str:
    """Fill {placeholders} in a template from the actual result files."""
    ctx = {}
    try:
        if rel in ("results/02_network_final_top30.json", "results/01_network_bootstrap_mean_top30_table1_input.json"):
            ctx = _json_edge_stats(os.path.join(ROOT, rel))
        elif rel == "results/02_threshold_sensitivity.csv":
            import csv as _csv
            with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
                rows = list(_csv.DictReader(fh))
            r70 = next((r for r in rows if r["percentile"] == "70"), None)
            if r70:
                ctx = {"thr70": r70["threshold"], "e70": r70["n_edges"]}
        elif rel == "results/TableS3.csv":
            import csv as _csv
            with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
                rows = list(_csv.DictReader(fh))

            def _b(x):
                return str(x).strip().lower() in ("true", "1")

            # classification (genuine/borderline/artefact) is only defined for
            # rows that are negative edges in the reported network; restrict
            # both the count and the breakdown to that subset, matching
            # negative_edges_analysis.py's own reporting exactly.
            net_rows = [r for r in rows if _b(r.get("in_final_network_negative_edge"))]
            n_neg = len(net_rows)
            genuine = sum(1 for r in net_rows if r.get("classification") == "genuine")
            borderline = sum(1 for r in net_rows if r.get("classification") == "borderline")
            artefact = sum(1 for r in net_rows if r.get("classification") == "artefact")
            ctx = dict(n_neg=n_neg, genuine=genuine, borderline=borderline, artefact=artefact)
        elif rel == "results/TableS9.csv":
            import csv as _csv
            with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
                rows = list(_csv.DictReader(fh))
            ctx = {"k": len({r["cluster"] for r in rows})}
        elif rel == "results/ground_truth_validation/TableS5.csv":
            ctx = {"vae_pairs": "phi_raw / l1_signed / l1_logistic_bagged / bootstrap_mean"}
        elif rel in ("results/01_jaccard_differential_summary.csv", "results/01_jaccard_differential_edges.csv"):
            import csv as _csv
            with open(os.path.join(ROOT, "results/01_jaccard_differential_summary.csv"), encoding="utf-8") as fh:
                row = next(_csv.DictReader(fh))
            ctx = dict(bm=row["bm_top30_edges"], jac=row["jaccard_top30_edges"],
                       bm_only=row["bm_only_edges"], jac_only=row["jaccard_only_edges"])
    except Exception as exc:  # pragma: no cover - manifest must not crash
        return template + f"  [note: could not fill placeholders: {exc}]"
    try:
        return template.format(**ctx)
    except (KeyError, IndexError):
        return template


def main() -> int:
    if not os.path.isdir(RESULTS):
        raise SystemExit("results/ does not exist yet - run the pipeline first.")

    files = []
    for dirpath, _dirs, names in os.walk(RESULTS):
        for name in names:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
            files.append((rel, full))
    files.sort()

    rows = []
    for rel, full in files:
        st = os.stat(full)
        template = CIFRA.get(rel, "")
        cifra = _fill(rel, template) if template else ""
        rows.append({
            "archivo": os.path.basename(rel),
            "ruta_relativa": rel,
            "sha256": sha256(full),
            "bytes": st.st_size,
            "fecha_generacion": dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "cifra_clave": cifra,
        })

    with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["archivo", "ruta_relativa", "sha256",
                                           "bytes", "fecha_generacion", "cifra_clave"])
        w.writeheader()
        w.writerows(rows)

    print(f"MANIFEST.csv written: {len(rows)} files under results/")
    missing = [r["ruta_relativa"] for r in rows if not r["cifra_clave"]]
    if missing:
        print(f"  ({len(missing)} files without a curated key figure:)")
        for m in missing:
            print(f"    {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
