"""
Cross-check of the minimum number of changes in ancestral_reconstruction.py against an independent
implementation (Bio.Phylo's ParsimonyScorer, Fitch).  Same tree (midpoint-rooted in memory); the
Fitch score does not depend on the root for a binary tree.  Exits with status 1 on any mismatch so
that run_all.py stops.

Output: results/phylogenetic_sensitivity/ancestral_reconstruction/fitch_validation.txt
"""
import sys

from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import ParsimonyScorer
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ancestral_common import OUT_DIR, load_matrix, load_rooted_tree
from ancestral_reconstruction import fitch


def main() -> int:
    M = load_matrix()
    t = load_rooted_tree(M.columns)
    lines = ["Fitch validation: own implementation (src/ancestral_reconstruction.py) vs Bio.Phylo ParsimonyScorer",
             "tree: 06_nj_tree_150strains.nwk, midpoint-rooted in memory (150 tips); all 63 effector families", ""]
    ok = True
    for e in M.index:
        g, l, _ = fitch(t, M.loc[e], 1)
        aln = MultipleSeqAlignment([SeqRecord(Seq("A" if M.loc[e, s] else "C"), id=s) for s in M.columns])
        biop = int(ParsimonyScorer().get_score(t, aln))
        ok &= biop == g + l
        lines.append(f"{e:12s} own={g + l:3d}  Bio.Phylo={biop:3d}  {'MATCH' if biop == g + l else 'MISMATCH'}")
    lines += ["", f"{len(M.index)} effector families compared",
              "ALL MATCH" if ok else "MISMATCH FOUND"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fitch_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
