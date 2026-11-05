"""ML ancestral reconstruction with PastML (MPPA, F81) for NleG/NleD/EspM/EspM1 on three rootings:
  midpoint   - same root as the parsimony run
  minvar     - root minimising the variance of root-to-tip distances (best position over all edges)
  inner      - root at the midpoint of the edge above the larger child of the midpoint root's non-basal side
               (a deliberately different, deeper-in-the-main-clade root, to stress the root dependence)
Expected gains/losses come from the MARGINAL node probabilities per branch:
  E[gain] = sum_edges P_parent(0)*P_child(1),  E[loss] = sum_edges P_parent(1)*P_child(0).
This ignores >1 change per branch, but the product of marginals is not the joint branch posterior, so at ambiguous
nodes it ADDS spurious events (can overcount). map_* columns count changes between the most probable states.
OPTIONAL, NOT part of run_all.py: needs its own environment (Python 3.12 + pastml, ete3, six, biopython;
see extras/pastml/README.md) because PastML pins a scipy that conflicts with requirements.txt.
Input tree and rooting come from src/ancestral_common.py (06_nj_tree_150strains.nwk, midpoint-rooted in memory),
so the root is identical to the parsimony run.  Output: extras/pastml/output/events_ml.csv."""
import sys
from io import StringIO
from pathlib import Path

import numpy as np, pandas as pd
from Bio import Phylo
from ete3 import Tree
from pastml.acr import acr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from ancestral_common import load_matrix, load_rooted_tree  # noqa: E402

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
M = load_matrix()
_buf = StringIO(); Phylo.write(load_rooted_tree(M.columns), _buf, "newick")
base = Tree(_buf.getvalue(), format=1)

def minvar_edge(t):
    """(variance, node name, position x from parent along the edge, edge length, n tips below)."""
    tips = t.get_leaves(); best = None
    for c in t.traverse():
        if c.is_root(): continue
        L, p = c.dist, c.up
        side = set(c.get_leaves())
        slope = np.array([-1.0 if x in side else 1.0 for x in tips])
        icpt = np.array([c.get_distance(x) + L if x in side else p.get_distance(x) for x in tips])
        xs = np.linspace(0, L, 201) if L > 0 else np.array([0.0])
        v = [np.var(slope * xx + icpt) for xx in xs]; i = int(np.argmin(v))
        if best is None or v[i] < best[0]: best = (v[i], c.name, xs[i], L, len(side))
    return best

def reroot(t, cname, x, L):
    """Root at distance x from the parent end of the edge above `cname` (L = its length)."""
    t = t.copy(); c = t.search_nodes(name=cname)[0]
    t.set_outgroup(c)
    for k in t.get_tree_root().children:
        k.dist = (L - x) if k is c else x
    for n in list(t.traverse()):
        if not n.is_root() and not n.is_leaf() and len(n.children) == 1: n.delete(preserve_branch_length=True)
    return t

mv = minvar_edge(base)
print("minvar root: edge above %s (%d tips below), x=%.6f of L=%.6f" % (mv[1], mv[4], mv[2], mv[3]))
big = max(base.get_tree_root().children, key=lambda k: len(k))
inner = max(big.children, key=lambda k: len(k))
print("inner root: edge above %s (%d tips below), midpoint of %.6f" % (inner.name, len(inner), inner.dist))
trees = {"midpoint": base,
         "minvar": reroot(base, mv[1], mv[2], mv[3]),
         "inner": reroot(base, inner.name, inner.dist / 2, inner.dist)}

rows = []
for rname, T in trees.items():
    for e in ["NleG", "NleD", "EspM", "EspM1"]:
        df = pd.DataFrame({e: M.loc[e].astype(str)}); df.index.name = "id"
        tt = T.copy()
        r = acr(tt, df, prediction_method="MPPA", model="F81")[0]
        mp = r["marginal_probabilities"]
        p1 = lambda n: float(mp.loc[n.name, "1"])
        g = l = 0.0; gm = lm = 0
        for n in tt.traverse():
            if n.is_root(): continue
            pp, pc = p1(n.up), p1(n)
            g += (1 - pp) * pc; l += pp * (1 - pc)
            gm += int(pp < 0.5 <= pc); lm += int(pp >= 0.5 > pc)   # hard calls at the most probable state
        rows.append(dict(effector=e, root=rname, method="PastML MPPA F81", exp_gains_marginal=round(g, 3),
                         exp_losses_marginal=round(l, 3), exp_total_marginal=round(g + l, 3),
                         map_gains=gm, map_losses=lm, map_total=gm + lm,
                         root_P1=round(p1(tt.get_tree_root()), 4), loglik=round(float(r["log_likelihood"]), 3),
                         n_unresolved_nodes=int(r["num_unresolved_nodes"])))
ml = pd.DataFrame(rows); ml.to_csv(OUT / "events_ml.csv", index=False); print(ml.to_string())
