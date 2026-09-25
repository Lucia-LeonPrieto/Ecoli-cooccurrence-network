"""
Negative-edge interpretability analysis (Table S3).

The Benjamini-Hochberg (BH) correction must be applied over the full
universe of candidate effector pairs, not only over the negative edges
already present in the final thresholded network -- restricting the
correction to a small subset selected post hoc by the network threshold
would understate the multiple-testing burden.

The 2x2 contingency table and the one-sided Fisher exact test
(alternative="less") are computed for ALL 1953 pairs of effectors, BH
correction (statsmodels.stats.multitest.multipletests) is applied to the
FULL vector of 1953 p-values, and ONLY THEN is the corrected table filtered
down to the pairs that are negative edges in the final thresholded network,
for the genuine/borderline/artefact classification reported in the
manuscript. The output CSV keeps ALL 1953 pairs (not just the network's
negative edges) with an explicit `in_final_network_negative_edge` column, so
a reader can see both how many of the 1953 candidate pairs validate in
general and how many of those coincide with edges present in the final
network.

Inputs  (relative to the repository root):
    data/input/EffectorRepertoire_Ecoli.xlsx   -> binary matrix (Block 1 logic)
    results/02_network_final_top30.json            -> the thresholded network
                                                    (used only to flag which
                                                    of the 1953 pairs are
                                                    negative edges in it)
Output:
    results/TableS3.csv   -> ALL 1953 pairs, BH-corrected over the full
                              universe (manuscript's Table S3)

Usage:
    python src/negative_edges_analysis.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from inference_functions import load_binary_matrix  # noqa: E402

# ── configuration ────────────────────────────────────────────────────────────
INPUT_XLSX = ROOT / "data" / "input" / "EffectorRepertoire_Ecoli.xlsx"
JSON_PATH = ROOT / "results" / "02_network_final_top30.json"
OUT_PATH = ROOT / "results" / "TableS3.csv"

NEG_THRESHOLD = 0.0   # analyse every edge with weight < this value
TOP_N = 20            # rows printed to stdout (the CSV is always complete)
# ─────────────────────────────────────────────────────────────────────────────


def contingency(matrix, eff_a, eff_b):
    if eff_a not in matrix.index or eff_b not in matrix.index:
        return None

    a = matrix.loc[eff_a].values.astype(int)
    b = matrix.loc[eff_b].values.astype(int)
    n = len(a)

    n_both = int(np.sum((a == 1) & (b == 1)))
    n_a_only = int(np.sum((a == 1) & (b == 0)))
    n_b_only = int(np.sum((a == 0) & (b == 1)))
    n_neither = int(np.sum((a == 0) & (b == 0)))

    prev_a = (n_both + n_a_only) / n
    prev_b = (n_both + n_b_only) / n
    expected = prev_a * prev_b * n
    lift = (n_both / n) / (prev_a * prev_b) if (prev_a * prev_b) > 0 else np.nan

    table = np.array([[n_both, n_a_only], [n_b_only, n_neither]])
    odds_ratio, p_value = fisher_exact(table, alternative="less")

    return {
        "n_both": n_both, "n_a_only": n_a_only, "n_b_only": n_b_only,
        "n_neither": n_neither, "prev_a": prev_a, "prev_b": prev_b,
        "expected": expected, "lift": lift, "odds_ratio": odds_ratio,
        "p_value": p_value,
    }


def main() -> int:
    matrix = load_binary_matrix(INPUT_XLSX)
    n_total = matrix.shape[1]
    effectors = matrix.index.tolist()

    print("\n" + "=" * 65)
    print("  NEGATIVE EDGE INTERPRETABILITY ANALYSIS")
    print("=" * 65)
    print(f"  effectors : {matrix.shape[0]}")
    print(f"  strains   : {n_total}")

    with open(JSON_PATH, encoding="utf-8") as fh:
        net = json.load(fh)

    # Negative edges actually present in the final thresholded network --
    # used ONLY to flag membership (see `in_final_network_negative_edge`
    # below); scoring/testing/correction below runs over ALL 1953 pairs.
    net_neg_edge_scores = {
        tuple(sorted((e["data"]["source"], e["data"]["target"]))): e["data"]["weight"]
        for e in net["elements"]["edges"]
        if e["data"]["weight"] < NEG_THRESHOLD
    }
    print(f"\n  negative edges in final network : {len(net_neg_edge_scores)}")

    # Iterate over ALL C(63,2) = 1953 candidate pairs, not just the negative
    # edges already in the final network.
    all_pairs = list(combinations(effectors, 2))
    print(f"  total candidate pairs (all effectors) : {len(all_pairs)}")

    results = []
    for eff_a, eff_b in all_pairs:
        c = contingency(matrix, eff_a, eff_b)
        if c is None:  # pragma: no cover - defensive, should never trigger
            print(f"  WARNING: {eff_a} or {eff_b} not found in matrix")
            continue
        n_a = c["n_both"] + c["n_a_only"]
        n_b = c["n_both"] + c["n_b_only"]
        min_overlap = max(0, n_a + n_b - n_total)
        key = tuple(sorted((eff_a, eff_b)))
        in_final_neg = key in net_neg_edge_scores
        c.update({
            "eff_a": eff_a, "eff_b": eff_b,
            "score": net_neg_edge_scores.get(key, np.nan),
            "n_a": n_a, "n_b": n_b, "min_overlap": min_overlap,
            "in_final_network_negative_edge": in_final_neg,
        })
        results.append(c)

    assert len(results) == 1953, f"expected C(63,2)=1953 pairs, got {len(results)}"

    # BH correction over the FULL 1953-pair universe.
    p_values = [c["p_value"] for c in results]
    reject, p_values_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    for c, p_adj, is_rejected in zip(results, p_values_adj, reject):
        c["p_Fisher_adj"] = p_adj
        c["coexclusion_validated"] = bool(c["lift"] < 1 and is_rejected)

    lift_lt_1_all = sum(c["lift"] < 1 for c in results)
    validated_bh_all = sum(c["coexclusion_validated"] for c in results)
    print(f"\n  [full universe, n=1953]")
    print(f"  pairs with lift < 1                : {lift_lt_1_all}")
    print(f"  validated after BH (p_adj < 0.05)  : {validated_bh_all}")
    print(f"  lost after correction              : {lift_lt_1_all - validated_bh_all}")

    # Manuscript reporting subset: only the pairs that are negative edges in
    # the final thresholded network, using the p_adj computed over the FULL
    # 1953-pair universe above (not recomputed on this subset).
    net_results = [c for c in results if c["in_final_network_negative_edge"]]
    net_results.sort(key=lambda c: c["score"])
    lift_lt_1_net = sum(c["lift"] < 1 for c in net_results)
    validated_bh_net = sum(c["coexclusion_validated"] for c in net_results)
    print(f"\n  [final-network negative edges only, n={len(net_results)}, "
          f"p_adj from the full 1953-pair correction]")
    print(f"  pairs with lift < 1                : {lift_lt_1_net}")
    print(f"  validated after BH (p_adj < 0.05)  : {validated_bh_net}")
    print(f"  lost after correction              : {lift_lt_1_net - validated_bh_net}")

    # genuine: lift < 0.8 AND validated after BH
    # borderline: lift < 1 AND not validated
    # artefact: everything else (lift >= 1, or the residual edge cases)
    # Classification restricted to the final-network negative edges (as
    # before), but now driven by the full-universe-corrected p_adj.
    genuine, borderline, artefact = [], [], []
    for c in net_results:
        if c["lift"] < 0.8 and c["coexclusion_validated"]:
            c["classification"] = "genuine"
            genuine.append(c)
        elif c["lift"] < 1.0 and not c["coexclusion_validated"]:
            c["classification"] = "borderline"
            borderline.append(c)
        else:
            c["classification"] = "artefact"
            artefact.append(c)
    print(f"\n  genuine exclusion (lift < 0.8, BH p < 0.05) : {len(genuine)}")
    print(f"  borderline (lift < 1, not validated)        : {len(borderline)}")
    print(f"  likely artefact (else)                      : {len(artefact)}")

    header = (f"\n  {'Pair':<30} {'Score':>7} {'Both':>5} {'A_only':>7} "
              f"{'B_only':>7} {'Exp':>6} {'Lift':>6} {'p_Fisher':>10}")
    print(header)
    print(f"  {'-' * 80}")
    for c in net_results[:TOP_N]:
        pair = f"{c['eff_a']} - {c['eff_b']}"
        print(f"  {pair:<30} {c['score']:>7.3f} {c['n_both']:>5} "
              f"{c['n_a_only']:>7} {c['n_b_only']:>7} "
              f"{c['expected']:>6.1f} {c['lift']:>6.3f} {c['p_value']:>10.4f}")

    # classification (genuine/borderline/artefact) is only defined for the
    # final-network negative edges above; every other pair (not a negative
    # edge in the reported network) gets "" -- it was never a candidate for
    # this classification in the manuscript.
    for c in results:
        c.setdefault("classification", "")

    # Output: ALL 1953 pairs (point 6), sorted by raw p-value for readability,
    # with in_final_network_negative_edge distinguishing the manuscript's
    # reporting subset.
    df_out = pd.DataFrame(results).rename(columns={"p_value": "p_Fisher"})[[
        "eff_a", "eff_b", "score", "n_a", "n_b", "n_both",
        "n_a_only", "n_b_only", "n_neither", "expected",
        "lift", "odds_ratio", "p_Fisher", "p_Fisher_adj",
        "coexclusion_validated", "in_final_network_negative_edge",
        "classification", "min_overlap",
    ]].sort_values("p_Fisher")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}  ({len(df_out)} rows, "
          f"{df_out['in_final_network_negative_edge'].sum()} flagged as final-network negative edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
