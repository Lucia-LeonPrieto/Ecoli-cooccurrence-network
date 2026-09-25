#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cmp_saga_liblinear.py
======================
The saga-vs-liblinear solver comparison for the L1-logistic bagged network,
cited in the manuscript (Methods 2.5).

For each of N_REP=25 bootstrap replicates (same resampled 60%-of-strains
subsample per replicate, drawn once from np.random.default_rng(SEED), shared
between solvers so the comparison is matched/paired rather than between two
independent baggings), fits neighborhood_selection_logistic with solver=
'saga' and solver='liblinear' on the identical subsample, then computes the
Pearson correlation and sign agreement between the two resulting (signed,
symmetrized) adjacency matrices' upper triangle (excluding the diagonal).
Reports the mean Pearson r and mean sign agreement across the 25 replicates.

Outputs:
    logs/cmp_saga_liblinear_25rep.out       human-readable summary (this run)
    results/ground_truth_validation/saga_vs_liblinear_25rep.csv   per-replicate detail

Wired into run_all.py as step 13; independent of every other step. Run
standalone with:
    python src/cmp_saga_liblinear.py
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from inference_functions import load_binary_matrix, neighborhood_selection_logistic  # noqa: E402

INPUT_XLSX = ROOT / "data" / "input" / "EffectorRepertoire_Ecoli.xlsx"
LOG_OUT = ROOT / "logs" / "cmp_saga_liblinear_25rep.out"
CSV_OUT = ROOT / "results" / "ground_truth_validation" / "saga_vs_liblinear_25rep.csv"

N_REP = 25
SAMPLE_FRACTION = 0.6
SEED = 42


def upper_tri(mat):
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]


def main():
    warnings.filterwarnings("ignore")
    t0 = time.time()
    matrix = load_binary_matrix(INPUT_XLSX)
    n_samples = matrix.shape[1]
    size = int(SAMPLE_FRACTION * n_samples)

    rng = np.random.default_rng(SEED)
    idx_per_rep = [rng.choice(n_samples, size=size, replace=True) for _ in range(N_REP)]

    rows = []
    for b, idx in enumerate(idx_per_rep):
        sub = matrix.iloc[:, idx]

        np.random.seed(SEED + b)
        adj_saga, _ = neighborhood_selection_logistic(
            sub, normalize_adj=False, symm_rule="average_signed", solver="saga",
        )
        np.random.seed(SEED + b)
        adj_lib, _ = neighborhood_selection_logistic(
            sub, normalize_adj=False, symm_rule="average_signed", solver="liblinear",
        )

        v_saga = upper_tri(adj_saga.values)
        v_lib = upper_tri(adj_lib.values)
        r = np.corrcoef(v_saga, v_lib)[0, 1]
        sign_agree = np.mean(np.sign(v_saga) == np.sign(v_lib))

        rows.append(dict(replicate=b, pearson_r=r, sign_agreement=sign_agree))
        print(f"  replicate {b + 1:02d}/{N_REP}: r={r:.4f}  sign_agreement={sign_agree:.4f}  "
              f"elapsed={time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)

    mean_r = df["pearson_r"].mean()
    mean_sign = df["sign_agreement"].mean()

    summary = (
        f"saga vs liblinear -- L1-logistic bagged solver comparison\n"
        f"{'=' * 60}\n"
        f"N replicates       : {N_REP}\n"
        f"sample_fraction     : {SAMPLE_FRACTION}\n"
        f"seed                : {SEED}\n"
        f"mean Pearson r      : {mean_r:.4f}\n"
        f"mean sign agreement : {mean_sign:.4f}\n"
        f"total runtime       : {time.time() - t0:.1f}s\n"
        f"\nPer-replicate detail in {CSV_OUT}\n"
    )
    print("\n" + summary)
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_OUT, "w", encoding="utf-8") as fh:
        fh.write(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
