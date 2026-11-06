"""
Per-lineage counts for the two effector pairs discussed by the reviewers (NleG/NleD and EspM/EspM1).

For each pair, for each of the 5 UPGMA lineages (results/phylogenetic_sensitivity/06_lineage_clusters.csv)
and for both subsets (strains INSIDE the lineage, and all strains EXCEPT that lineage), report:
    N strains, n carriers of each effector, n with both, expected co-carriers under independence
    (n_a * n_b / N), P(X <= observed) from the hypergeometric distribution (exclusion-side test:
    "this few co-carriers or fewer by chance"), and lift (= observed / expected, NaN if expected = 0).

Where an effector is absent from the subset (expected = 0) the test is uninformative: P = 1 and lift = NaN.

Output: results/phylogenetic_sensitivity/pair_lineage_counts.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parent.parent
MATRIX_CSV = ROOT / "results" / "01_binary_matrix.csv"
LINEAGE_CSV = ROOT / "results" / "phylogenetic_sensitivity" / "06_lineage_clusters.csv"
OUT_CSV = ROOT / "results" / "phylogenetic_sensitivity" / "pair_lineage_counts.csv"

PAIRS = [("NleG", "NleD"), ("EspM", "EspM1")]


def counts(M, a, b, strains):
    x, y = M.loc[a, strains], M.loc[b, strains]
    N, na, nb, both = len(strains), int(x.sum()), int(y.sum()), int((x & y).sum())
    exp = na * nb / N if N else np.nan
    return dict(N=N, **{f"n_{a}": na, f"n_{b}": nb}, n_both=both, expected=round(exp, 4),
                p_hypergeom_le=hypergeom.cdf(both, N, na, nb) if N else np.nan,
                lift=round(both / exp, 4) if exp > 0 else np.nan)


def main():
    M = pd.read_csv(MATRIX_CSV, index_col=0)
    lc = pd.read_csv(LINEAGE_CSV)
    assert set(lc.strain) == set(M.columns), "lineage assignment != matrix columns"
    rows = []
    for a, b in PAIRS:
        for k in sorted(lc.lineage_cluster.unique()):
            inside = list(lc.loc[lc.lineage_cluster == k, "strain"])
            outside = list(lc.loc[lc.lineage_cluster != k, "strain"])
            for subset, strains in (("within", inside), ("excluded", outside)):
                c = counts(M, a, b, strains)
                rows.append(dict(pair=f"{a}/{b}", eff_a=a, eff_b=b, lineage=k, subset=subset,
                                 N=c["N"], n_a=c[f"n_{a}"], n_b=c[f"n_{b}"], n_both=c["n_both"],
                                 expected=c["expected"], p_hypergeom_le=c["p_hypergeom_le"], lift=c["lift"]))
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"  -> {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
