"""
Fitch-parsimony ancestral reconstruction and gain/loss counts for the two effector pairs
NleG/NleD and EspM/EspM1, on the single NJ tree
06_nj_tree_150strains.nwk (midpoint-rooted in memory, see ancestral_common.py).

Parsimony gives the MINIMUM number of changes.  Where a node is ambiguous the Fitch resolution is
run twice (ambiguity resolved towards presence / towards absence); both are reported because the
gain/loss split (not the total) can depend on it.

Minimum transitions are also computed for all 63 effector families (Table S14).

Outputs (results/phylogenetic_sensitivity/ancestral_reconstruction/):
    events.csv                  one row per effector (the 4 effectors of the two pairs)
    TableS14.csv                all 63 families: carriers and minimum transitions
    TableS14.xlsx               same table + a "Notes" sheet with the legend
    tree_NleG_NleD.png          tips coloured by presence of each effector of the pair
    tree_EspM_EspM1.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from Bio import Phylo

from ancestral_common import EFFECTORS, OUT_DIR, PAIRS, load_matrix, load_rooted_tree


def fitch(t, state, prefer):
    """Fitch parsimony on rooted binary tree `t`; `state` maps tip name -> 0/1.
    Returns (n_gain, n_loss, node_state) with ambiguity resolved towards `prefer`."""
    sets = {}

    def down(c):
        if c.is_terminal():
            sets[c] = {int(state[c.name])}
            return
        for k in c.clades:
            down(k)
        a = [sets[k] for k in c.clades]
        inter = set.intersection(*a)
        sets[c] = inter if inter else set.union(*a)

    down(t.root)
    st = {}

    def up(c, parent):
        s = sets[c]
        if parent is not None and parent in s:
            st[c] = parent
        elif prefer in s:
            st[c] = prefer
        else:
            st[c] = min(s)
        for k in c.clades:
            up(k, st[c])

    up(t.root, None)
    g = l = 0
    for c in t.find_clades():
        for k in c.clades:
            if st[c] == 0 and st[k] == 1:
                g += 1
            if st[c] == 1 and st[k] == 0:
                l += 1
    return g, l, st


def count_events(t, M):
    rows = []
    for e in EFFECTORS:
        s = M.loc[e]
        g1, l1, st1 = fitch(t, s, 1)
        g0, l0, st0 = fitch(t, s, 0)
        rows.append(dict(effector=e, n_strains=int(s.sum()), method="Fitch parsimony (midpoint-rooted)",
                         gains_pref1=g1, losses_pref1=l1, gains_pref0=g0, losses_pref0=l0,
                         total_transitions=g1 + l1, root_state_pref1=int(st1[t.root]),
                         root_state_pref0=int(st0[t.root])))
    return pd.DataFrame(rows)


def table_s14(t, M):
    """Minimum number of transitions (gains + losses, Fitch) for every effector family.
    Sorted by ascending transitions, ties by name.  A family present in all or in no strain has
    0 transitions (a constant character needs no change on the tree)."""
    rows = []
    for e in M.index:
        g, l, _ = fitch(t, M.loc[e], 1)
        rows.append({"Effector family": e, "Strains carrying it": int(M.loc[e].sum()),
                     "Minimum transitions": g + l})
    return (pd.DataFrame(rows)
            .sort_values(["Minimum transitions", "Effector family"], kind="mergesort")
            .reset_index(drop=True))


S14_LEGEND = (
    "Minimum number of state transitions per effector family. Ancestral states were reconstructed "
    "by parsimony (Fitch, 1971, Systematic Zoology 20:406-416) on the neighbour-joining tree built "
    "from the genomic distance matrix, rooted at its midpoint and without outgroup, and the counts "
    "were confirmed with an independent implementation (Biopython). Because the tree derives from "
    "genomic distances, counts are minimum estimates."
)


def write_s14_xlsx(s14, path):
    from openpyxl.styles import Alignment, Font
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        s14.to_excel(xw, sheet_name="Table S14", index=False)
        ws = xw.sheets["Table S14"]
        for c in ws[1]:
            c.font = Font(bold=True)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for col, w in zip("ABC", (22, 20, 22)):
            ws.column_dimensions[col].width = w
        ws.freeze_panes = "A2"
        pd.DataFrame({"Notes": [S14_LEGEND]}).to_excel(xw, sheet_name="Notes", index=False)
        wn = xw.sheets["Notes"]
        wn["A1"].font = Font(bold=True)
        wn["A2"].alignment = Alignment(wrap_text=True, vertical="top")
        wn.column_dimensions["A"].width = 110


def draw_pair(t, M, a, b, path):
    tips = t.get_terminals()
    fig, ax = plt.subplots(figsize=(8, 12))
    Phylo.draw(t, axes=ax, do_show=False, label_func=lambda c: "", show_confidence=False)
    ys = {c.name: i + 1 for i, c in enumerate(tips)}
    xmax = max(t.depths().values())
    cols = {(0, 0): "#cccccc", (1, 0): "#d95f02", (0, 1): "#1b9e77", (1, 1): "#7570b3"}
    for c in tips:
        k = (int(M.loc[a, c.name]), int(M.loc[b, c.name]))
        ax.plot(xmax * 1.02, ys[c.name], "s", color=cols[k], ms=4)
    for k, lab in [((1, 0), f"{a} only"), ((0, 1), f"{b} only"), ((1, 1), "both"), ((0, 0), "neither")]:
        ax.plot([], [], "s", color=cols[k], label=lab)
    ax.legend(loc="lower right")
    ax.set_title(f"{a} vs {b} (midpoint-rooted NJ tree, 150 strains)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    M = load_matrix()
    t = load_rooted_tree(M.columns)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev = count_events(t, M)
    ev.to_csv(OUT_DIR / "events.csv", index=False)
    print(ev.to_string())
    print(f"  -> {(OUT_DIR / 'events.csv').relative_to(OUT_DIR.parents[2])}")
    s14 = table_s14(t, M)
    s14.to_csv(OUT_DIR / "TableS14.csv", index=False)
    print(f"  -> {(OUT_DIR / 'TableS14.csv').relative_to(OUT_DIR.parents[2])}  ({len(s14)} effector families)")
    write_s14_xlsx(s14, OUT_DIR / "TableS14.xlsx")
    for a, b in PAIRS:
        draw_pair(t, M, a, b, OUT_DIR / f"tree_{a}_{b}.png")


if __name__ == "__main__":
    main()
