#!/usr/bin/env python
"""
run_all.py — reproduce the entire analysis from a clean checkout.

    git clone <repo>
    cd ecoli-effector-network
    python -m venv .venv && .venv/Scripts/activate     # or source .venv/bin/activate
    pip install -r requirements.txt
    python run_all.py

No paths to edit.  Every script reads from data/ and writes to results/.

Steps, in order:
    1. src/pipeline_main.py                    intermediates + Bootstrap-Mean network + Table 1
    2. src/build_final_network.py              the reported network + per-edge confidence
    3. src/negative_edges_analysis.py          Table S3 (negative-edge interpretability,
                                                BH-corrected over all 1953 candidate pairs)
    4. src/hierarchical_clustering.py          k = 7 modules + dendrogram
    5. src/simulation_ground_truth.py          known-truth validation (7 methods, 35 reps)
    6. src/phylogenetic_sensitivity.py         leave-one-lineage-out
    7. src/effector_lineage_distribution.py    per-lineage carrier distribution for all
                                                63 effectors (Reviewer-1 lineage-confound
                                                check); depends on step 6's lineage
                                                assignment, so it must run right after it
    8. src/edge_sign_stability_lineage.py      VAE-based: for each of the 585 reported
                                                edges, does the decoder-cosine weight (sign
                                                AND continuous magnitude) survive each
                                                lineage exclusion; reuses step 6's cached
                                                single-shot reconstructions, no retraining
    9. src/edge_contingency_lineage_check.py   VAE-FREE: for the same 585 edges, does the
                                                raw 2x2-contingency lift/odds-ratio
                                                direction (not the VAE's output) survive
                                                each lineage exclusion; independent of the
                                                VAE entirely, only needs step 6's lineage
                                                assignment (results/phylogenetic_sensitivity/
                                                lineage_clusters.csv) and the binary matrix.
                                                Also writes a full-150-strain, no-exclusion
                                                baseline (same contingency() call, no lineage
                                                dropped) used to tell "already disagreed with
                                                the VAE in the full data" apart from "only
                                                disagrees once a lineage is excluded"
    10. src/held_out_loglik_beta_kl_comparison.py   Table S4: held-out ELBO / log-likelihood
                                                (5-fold CV) across beta_kl, cited in Methods
                                                2.4. Self-contained EXCEPT for 3 of its 5
                                                beta_kl rows (0.01/0.05/0.10), which are read
                                                from results/ground_truth_validation/
                                                held_out_evaluation.csv -- the output of the
                                                one-time 48-config hyperparameter grid search
                                                (simulation_ground_truth.py --phase
                                                select-hyperparams), which is NOT part of
                                                run_all.py and is not re-run here either
                                                (hyperparameter selection is already fixed in
                                                src/vae_config.json; re-running the full grid
                                                on every reproduction would add hours for no
                                                benefit). held_out_evaluation.csv therefore
                                                ships with the repo's results/ as a checked-in
                                                input to this step, not something run_all.py
                                                regenerates from scratch.
    11. src/threshold_sensitivity_ground_truth.py   Table S6: accuracy-vs-threshold
                                                sensitivity analysis (percentile sweep
                                                20-40%) cited in Methods 2.7. Fully
                                                self-contained -- regenerates its own
                                                synthetic ground truth from the same 35
                                                replicates as Table 1, no dependency on any
                                                other step's output files.
    12. src/phylo_corrected_convergence.py     Mantel-Haenszel phylogeny-corrected pairwise
                                                association + bootstrap-supported clustering
                                                (Reviewer 2, point 3, Methods 2.9), plus the
                                                L1-logistic-bagged diagnostic
                                                (--include-l1-diagnostic) that documents why
                                                Mantel-Haenszel was chosen over it (Table S12).
                                                Reuses step 6's 5-lineage UPGMA partition
                                                (results/phylogenetic_sensitivity/
                                                lineage_clusters.csv), so it must run after
                                                step 6. Also runs a 10,000-repetition
                                                label-permutation null test for the ARI
                                                (results/phylo_corrected_convergence/
                                                12_ari_permutation_test.csv).
    13. src/cmp_saga_liblinear.py              saga-vs-liblinear solver agreement for the
                                                L1-logistic bagged network (Methods 2.5),
                                                25 matched bootstrap replicates. Independent
                                                of every other step; only needs the raw
                                                input matrix.
    14. src/extend_tableS2_beta_boundary.py    Table S2's two boundary-probe rows
                                                (beta_kl = 0.005, 0.001, same architecture and
                                                ground-truth criterion as the 48-config grid),
                                                cited in Methods 2.4. Appends to the checked-in
                                                48-row grid (results/ground_truth_validation/
                                                TableS2.csv, from the one-time hyperparameter
                                                search above) and reuses step 10's Table S4 for
                                                the held-out log-likelihood columns, so it must
                                                run after step 10.

Runtime is dominated by steps 1, 2, 5 and 12 (VAE / bootstrap-heavy); expect
most of a day on CPU. Use  --only 3,4  to run a subset, or  --from 3  to
resume.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

STEPS = [
    ("pipeline_main",          [sys.executable, os.path.join(SRC, "pipeline_main.py")]),
    ("build_final_network",    [sys.executable, os.path.join(SRC, "build_final_network.py")]),
    ("negative_edges_analysis", [sys.executable, os.path.join(SRC, "negative_edges_analysis.py")]),
    ("hierarchical_clustering", [sys.executable, os.path.join(SRC, "hierarchical_clustering.py")]),
    ("simulation_ground_truth", [sys.executable, os.path.join(SRC, "simulation_ground_truth.py"), "--phase", "full"]),
    ("phylogenetic_sensitivity", [sys.executable, os.path.join(SRC, "phylogenetic_sensitivity.py")]),
    ("effector_lineage_distribution", [sys.executable, os.path.join(SRC, "effector_lineage_distribution.py")]),
    # VAE-based: reuses phylogenetic_sensitivity's cached single-shot
    # reconstructions (results/phylogenetic_sensitivity/consensus_excl_cluster*.npy);
    # must run after step 6, but does NOT retrain anything itself.
    ("edge_sign_stability_lineage", [sys.executable, os.path.join(SRC, "edge_sign_stability_lineage.py")]),
    # VAE-FREE: raw 2x2-contingency directional check, independent of the VAE
    # training block entirely -- only needs step 6's lineage_clusters.csv and
    # the binary presence/absence matrix. Kept here (not earlier in the
    # pipeline) purely because it shares that lineage assignment, not because
    # it has any dependency on the VAE steps.
    ("edge_contingency_lineage_check", [sys.executable, os.path.join(SRC, "edge_contingency_lineage_check.py")]),
    ("held_out_loglik_beta_kl_comparison", [sys.executable, os.path.join(SRC, "held_out_loglik_beta_kl_comparison.py")]),
    ("threshold_sensitivity_ground_truth", [sys.executable, os.path.join(SRC, "threshold_sensitivity_ground_truth.py")]),
    ("phylo_corrected_convergence", [sys.executable, os.path.join(SRC, "phylo_corrected_convergence.py"),
                                      "--n-boot-stability", "500", "--include-l1-diagnostic"]),
    ("cmp_saga_liblinear",      [sys.executable, os.path.join(SRC, "cmp_saga_liblinear.py")]),
    ("extend_tableS2_beta_boundary", [sys.executable, os.path.join(SRC, "extend_tableS2_beta_boundary.py")]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated step numbers to run (1-based), e.g. 3,4")
    ap.add_argument("--from", dest="from_step", type=int, default=1, help="resume from this step number")
    args = ap.parse_args()

    names = [n for n, _ in STEPS]
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
    else:
        wanted = set(range(args.from_step, len(STEPS) + 1))

    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLBACKEND", "Agg")
    # step 2 runs build_final_network.py on its own; tell step 1 not to repeat it
    env["SKIP_BUILD_FINAL_NETWORK"] = "1"

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    t0 = time.time()
    print("=" * 70)
    print("  RUN ALL  —  E. coli effector co-occurrence network")
    print("=" * 70)
    print(f"  python : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  steps  : {', '.join(f'{i+1}:{n}' for i, n in enumerate(names))}")
    print(f"  running: {sorted(wanted)}")
    print("=" * 70)

    for i, (name, cmd) in enumerate(STEPS, 1):
        if i not in wanted:
            print(f"\n--- [{i}/{len(STEPS)}] {name}  (skipped)")
            continue
        print(f"\n{'=' * 70}\n  [{i}/{len(STEPS)}] {name}\n{'=' * 70}", flush=True)
        ts = time.time()
        r = subprocess.run(cmd, cwd=ROOT, env=env)
        dt = (time.time() - ts) / 60
        if r.returncode != 0:
            print(f"\n!! step {i} ({name}) FAILED with exit code {r.returncode} after {dt:.1f} min")
            return r.returncode
        print(f"\n-- step {i} ({name}) done in {dt:.1f} min")

    print(f"\n{'=' * 70}\n  ALL DONE in {(time.time() - t0) / 3600:.2f} h\n{'=' * 70}")
    print("  Results are in results/.  Regenerate the checksums with:")
    print("      python src/make_manifest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
