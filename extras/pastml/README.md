# extras/pastml — optional maximum-likelihood ancestral states (NOT part of run_all.py)

`ancestral_ml_pastml.py` reconstructs the presence/absence of NleG, NleD, EspM and EspM1 with PastML
(MPPA, F81) on three rootings (midpoint, minimum root-to-tip variance, a deeper interior root) and
writes `output/events_ml.csv`.

It lives outside `run_all.py` and outside `results/` because PastML pins `scipy>=1.14` / `numpy 2.x`, which
conflict with the pins in the top-level `requirements.txt` (scipy 1.13.1, numpy 1.26.4). Use a separate
environment:

    conda create -n pastml312 -c conda-forge python=3.12 pip
    conda activate pastml312
    pip install -r extras/pastml/requirements-pastml.txt
    python extras/pastml/ancestral_ml_pastml.py

The tree and its midpoint rooting come from `src/ancestral_common.py`, i.e. the same
`results/phylogenetic_sensitivity/06_nj_tree_150strains.nwk` used by the parsimony step.

Caveat: `exp_*_marginal` columns multiply parent and child marginal probabilities per branch, which is
not the joint branch posterior and inflates counts at ambiguous nodes; `map_*` columns (changes between
the most probable states) are the cleaner reference. Stochastic mapping (R/phytools) is the proper
expected-count estimator and is not run here.
