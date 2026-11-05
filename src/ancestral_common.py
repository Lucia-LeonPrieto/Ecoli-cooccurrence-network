"""
Shared loaders for the ancestral-state analysis (ancestral_reconstruction.py, fitch_validation.py,
and the optional PastML scripts in extras/pastml/).

Single input tree: results/phylogenetic_sensitivity/06_nj_tree_150strains.nwk (written by
phylogenetic_sensitivity.py).  It is rooted at its midpoint IN MEMORY every time; no rooted
copy is ever written to disk, so every consumer sees exactly the same rooting.
"""
from pathlib import Path

import pandas as pd
from Bio import Phylo

ROOT = Path(__file__).resolve().parent.parent
TREE_NWK = ROOT / "results" / "phylogenetic_sensitivity" / "06_nj_tree_150strains.nwk"
MATRIX_CSV = ROOT / "results" / "01_binary_matrix.csv"
OUT_DIR = ROOT / "results" / "phylogenetic_sensitivity" / "ancestral_reconstruction"

EFFECTORS = ["NleG", "NleD", "EspM", "EspM1"]
PAIRS = [("NleG", "NleD"), ("EspM", "EspM1")]


def load_matrix():
    """63 x 150 binary matrix (rows = effector families, columns = strains)."""
    return pd.read_csv(MATRIX_CSV, index_col=0)


def load_rooted_tree(strains=None):
    """06_nj_tree_150strains.nwk, midpoint-rooted in memory; internal nodes named N0, N1, ...
    Zero-length branches are left as they are (parsimony does not use lengths)."""
    t = Phylo.read(str(TREE_NWK), "newick")
    if strains is not None:
        assert {x.name for x in t.get_terminals()} == set(strains), "tree tips != matrix columns"
    t.root_at_midpoint()
    t.root.branch_length = 0
    for i, c in enumerate(t.get_nonterminals()):
        c.name = f"N{i}"
    return t
