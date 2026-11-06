"""
Ground-truth validation of the network-inference methods.
=====================================================================

Comprueba si los métodos de inferencia de red que ya usa el pipeline
(VAE de un disparo, Bootstrap Mean, Phi crudo y L1 con signo) recuperan
BORDES DIRECTOS y NO marcan como borde:
  - relaciones puramente INDIRECTAS  (cadena A-C-B, sin borde A-B)
  - relaciones inducidas por un FACTOR LATENTE compartido H
  - pares de RUIDO sin ninguna dependencia

NO modifica el pipeline principal (pipeline_analysis.py) ni las funciones de
inferencia: las importa tal cual desde funciones_figures.py.

Dos fases con PAUSA OBLIGATORIA entre ellas:
  FASE 1 (por defecto)  -> genera 5 réplicas piloto (seeds 0-4) y reporta las
                           4 comprobaciones de sanidad de la simulación.
  FASE 2 (--phase 2)    -> SÓLO tras aprobación del piloto: evaluación completa
                           con 30-40 réplicas (seeds >= 100), AUC-PR y FPR en
                           indirectos / latentes por método.

Uso:
    python simulation_ground_truth.py            # Fase 1 (piloto)
    python simulation_ground_truth.py --phase 2  # Fase 2 (tras aprobación)
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

try:                                    # consola Windows -> forzar UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from inference_functions import generate_synthetic_ground_truth, load_binary_matrix
from vae_config_loader import load_vae_params

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)  # deterministic + faster on this small VAE

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
INPUT_XLSX = os.path.join(ROOT, "data", "input", "EffectorRepertoire_Ecoli.xlsx")
OUT_DIR = os.path.join(ROOT, "results", "ground_truth_validation")
os.makedirs(OUT_DIR, exist_ok=True)

# Parámetros de la simulación (compartidos por Fase 1 y Fase 2)
SIM_KWARGS = dict(
    n_nodes=30,
    n_samples=150,
    w_direct=0.72,
    w_chain=0.45,
    w_latent=0.36,
    structured_prev_band=(0.15, 0.85),
)

PILOT_SEEDS = list(range(5))            # 0..4
PHASE2_SEEDS = list(range(100, 135))    # 35 réplicas, disjuntas del piloto

# Block of seeds disjoint from both PILOT_SEEDS (0-4) and PHASE2_SEEDS
# (100-134, the block used to report method performance in the manuscript's
# Table 1 / ground-truth comparison). Used ONLY for VAE hyperparameter
# selection (run_hyperparameter_search_ground_truth(), below), so that the
# accuracy used to pick latent_dim/hidden_dim/beta_kl is never estimated on
# the same replicates used to report the winning method's performance --
# selecting hyperparameters and reporting performance on the same data
# would leak information between the two.
TUNING_SEEDS = list(range(200, 230))    # 30 réplicas, disjuntas de piloto y evaluación final

# Umbrales de las comprobaciones del piloto
PREV_TOL = 0.05          # |prev empírica - objetivo| por nodo
RATIO_MIN = 2.0          # direct debe ser >= 2x indirect y >= 2x latent (en |r|)
NONZERO_MIN = 0.05       # indirect y latent deben mostrar ALGO de correlación

# Check de ruido (Opción A, aprobada): en vez de un número fijo, se compara
# contra el valor esperado teórico de |r| de Pearson entre dos vectores
# INDEPENDIENTES de longitud n:  r ~ N(0, 1/(n-1))  =>  E[|r|] = sqrt(2/(pi*(n-1))).
# Para n=150 eso es ~0.0654, justo lo que salió en el piloto. Se admite un 25 %
# de margen sobre ese nulo analítico.
NULL_ABS_R = float(np.sqrt(2.0 / (np.pi * (SIM_KWARGS["n_samples"] - 1))))
NOISE_FACTOR = 1.25      # ruido "OK" si  |r|_medio <= NOISE_FACTOR * NULL_ABS_R

# Separación direct/indirect de la Fase 2 (Opción A, aprobada): NO se exige por
# réplica, sino sobre el PROMEDIO CONJUNTO de todas las réplicas de la Fase 2.
# Criterio fijado ANTES de ver los resultados de los métodos.
PHASE2_SEP_RATIO_MIN = 2.0


# ---------------------------------------------------------------------
def load_real_matrix():
    """The real 63 x 150 binary effector matrix (Block 1 logic, from the xlsx)."""
    return load_binary_matrix(INPUT_XLSX)


def pair_corr(bin_df, i, j):
    """Pearson (= phi para binario) entre las filas i y j de la matriz sintética."""
    a = bin_df.iloc[i].to_numpy(dtype=float)
    b = bin_df.iloc[j].to_numpy(dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(pearsonr(a, b)[0])


def category_corrs(bin_df, gt):
    """Devuelve dict categoría -> np.array de correlaciones marginales empíricas."""
    out = {}
    out["direct"] = np.array([pair_corr(bin_df, i, j) for i, j, _ in gt["direct"]])
    out["direct_pos"] = np.array(
        [pair_corr(bin_df, i, j) for i, j, s in gt["direct"] if s > 0]
    )
    out["direct_neg"] = np.array(
        [pair_corr(bin_df, i, j) for i, j, s in gt["direct"] if s < 0]
    )
    out["indirect"] = np.array([pair_corr(bin_df, i, j) for i, j in gt["indirect"]])
    out["latent"] = np.array([pair_corr(bin_df, i, j) for i, j in gt["latent"]])
    out["noise"] = np.array([pair_corr(bin_df, i, j) for i, j in gt["noise"]])
    return out


# =====================================================================
# FASE 1b — PILOTO
# =====================================================================
def run_pilot():
    real = load_real_matrix()
    print("=" * 78)
    print("  FASE 1b — PILOTO DE VERIFICACIÓN DE LA SIMULACIÓN")
    print("=" * 78)
    print(f"  Matriz real            : {real.shape[0]} efectores x {real.shape[1]} cepas")
    print(f"  Prevalencia real       : min={real.mean(axis=1).min():.3f} "
          f"mediana={real.mean(axis=1).median():.3f} max={real.mean(axis=1).max():.3f}")
    print(f"  Parámetros simulación  : {SIM_KWARGS}")
    print(f"  Seeds piloto           : {PILOT_SEEDS}")
    print("-" * 78)

    rows = []
    per_seed_corr = {}
    for seed in PILOT_SEEDS:
        bin_df, gt = generate_synthetic_ground_truth(
            seed=seed, real_matrix=real, **SIM_KWARGS
        )
        meta = gt["meta"]
        target = np.array(meta["target_prevalence"])
        emp = bin_df.mean(axis=1).to_numpy()
        dev = np.abs(emp - target)

        corr = category_corrs(bin_df, gt)
        per_seed_corr[seed] = corr

        m_direct = float(np.mean(np.abs(corr["direct"])))
        m_dpos = float(np.mean(np.abs(corr["direct_pos"])))
        m_dneg = float(np.mean(np.abs(corr["direct_neg"])))
        m_indir = float(np.mean(np.abs(corr["indirect"])))
        m_latent = float(np.mean(np.abs(corr["latent"])))
        m_noise = float(np.mean(np.abs(corr["noise"])))
        p95_noise = float(np.percentile(np.abs(corr["noise"]), 95))

        # firma: ¿coincide el signo empírico con el de la verdad conocida?
        sign_ok = sum(
            1 for (i, j, s) in gt["direct"]
            if np.sign(pair_corr(bin_df, i, j)) == np.sign(s)
        )

        rows.append(dict(
            seed=seed,
            prev_max_dev=float(dev.max()),
            prev_n_out=int((dev > PREV_TOL).sum()),
            pd_scale=meta["pd_scale"],
            min_eig=meta["precision_min_eig"],
            r_direct=m_direct, r_direct_pos=m_dpos, r_direct_neg=m_dneg,
            r_indirect=m_indir, r_latent=m_latent,
            r_noise=m_noise, r_noise_p95=p95_noise,
            ratio_dir_indir=(m_direct / m_indir if m_indir > 0 else np.inf),
            ratio_dir_latent=(m_direct / m_latent if m_latent > 0 else np.inf),
            sign_ok=sign_ok, n_direct=len(gt["direct"]),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "pilot_checks.csv"), index=False)

    # ---- Impresión detallada ---------------------------------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print("\n[estructura de una réplica] (idéntica salvo prevalencias/umbral por seed)")
    _, gt0 = generate_synthetic_ground_truth(seed=PILOT_SEEDS[0], real_matrix=real, **SIM_KWARGS)
    m0 = gt0["meta"]
    print(f"  nodos                 : {m0['n_nodes']}  "
          f"(estructurados={m0['n_struct_nodes']}, ruido={m0['n_noise_nodes']}, H latentes={m0['n_latent_H']})")
    print(f"  pares 'direct'         : {len(gt0['direct'])}  "
          f"(3 pares+ , 3 pares- , 3x2 bordes de cadena)")
    print(f"  pares 'indirect'       : {len(gt0['indirect'])}  (A-B de cada cadena)")
    print(f"  pares 'latent'         : {len(gt0['latent'])}   (3 por grupo x 2 grupos)")
    print(f"  pares 'noise'          : {len(gt0['noise'])}")
    print(f"  precisión simétrica    : {m0['precision_is_symmetric']}  "
          f"min autovalor={m0['precision_min_eig']:.4f}  pd_scale={m0['pd_scale']:.3f}")

    print("\n" + "-" * 78)
    print("COMPROBACIÓN 1 — prevalencia empírica dentro de ±0.05 de la objetivo (por nodo)")
    print("-" * 78)
    for r in rows:
        status = "OK" if r["prev_n_out"] == 0 else f"FALLA ({r['prev_n_out']} nodos fuera)"
        print(f"  seed {r['seed']}: max|Δprev| = {r['prev_max_dev']:.4f}   -> {status}")
    c1 = all(r["prev_n_out"] == 0 for r in rows)

    print("\n" + "-" * 78)
    print("COMPROBACIÓN 2 — |r| 'direct' vs 'indirect' / 'latent'  [ORIENTATIVA]")
    print("   (Opción A aprobada: el criterio VINCULANTE de separación direct/indirect")
    print("    se evalúa sobre el PROMEDIO CONJUNTO de las réplicas de la Fase 2,")
    print("    no por réplica. Aquí sólo se muestra por réplica como referencia.)")
    print("-" * 78)
    print(f"  {'seed':>4} | {'|r|dir':>7} {'|r|dir+':>8} {'|r|dir-':>8} | "
          f"{'|r|indir':>9} {'|r|lat':>7} | {'dir/indir':>9} {'dir/lat':>8}")
    for r in rows:
        print(f"  {r['seed']:>4} | {r['r_direct']:>7.3f} {r['r_direct_pos']:>8.3f} "
              f"{r['r_direct_neg']:>8.3f} | {r['r_indirect']:>9.3f} {r['r_latent']:>7.3f} | "
              f"{r['ratio_dir_indir']:>9.2f} {r['ratio_dir_latent']:>8.2f}")
    c2 = all(r["ratio_dir_indir"] >= RATIO_MIN and r["ratio_dir_latent"] >= RATIO_MIN
             for r in rows)

    print("\n" + "-" * 78)
    print("COMPROBACIÓN 3 — 'indirect' y 'latent' muestran ALGO de correlación (no cero)")
    print(f"                 (umbral mínimo de |r| medio = {NONZERO_MIN})")
    print("-" * 78)
    for r in rows:
        ok_i = r["r_indirect"] > NONZERO_MIN
        ok_l = r["r_latent"] > NONZERO_MIN
        print(f"  seed {r['seed']}: |r|indirect = {r['r_indirect']:.3f} "
              f"({'OK' if ok_i else 'DEMASIADO BAJO'}) | "
              f"|r|latent = {r['r_latent']:.3f} ({'OK' if ok_l else 'DEMASIADO BAJO'})")
    c3 = all(r["r_indirect"] > NONZERO_MIN and r["r_latent"] > NONZERO_MIN for r in rows)

    print("\n" + "-" * 78)
    noise_gate = NOISE_FACTOR * NULL_ABS_R
    print("COMPROBACIÓN 4 — pares 'noise' compatibles con independencia")
    print(f"                 nulo analítico E[|r|] = sqrt(2/(pi*(n-1))) = {NULL_ABS_R:.4f}  "
          f"(n={SIM_KWARGS['n_samples']})")
    print(f"                 se admite |r| medio <= {NOISE_FACTOR:g} x nulo = {noise_gate:.4f}")
    print("-" * 78)
    for r in rows:
        ok = r["r_noise"] <= noise_gate
        print(f"  seed {r['seed']}: |r|noise medio = {r['r_noise']:.4f}  "
              f"(nulo {NULL_ABS_R:.4f}, ratio {r['r_noise'] / NULL_ABS_R:.2f}x)  "
              f"p95 = {r['r_noise_p95']:.4f}  -> {'OK' if ok else 'ALTO'}")
    c4 = all(r["r_noise"] <= noise_gate for r in rows)

    print("\n" + "-" * 78)
    print("DIAGNÓSTICO EXTRA — signo empírico vs verdad conocida en pares 'direct'")
    print("-" * 78)
    for r in rows:
        print(f"  seed {r['seed']}: {r['sign_ok']}/{r['n_direct']} pares con signo correcto")

    # ---- Resumen agregado ----------------------------------------
    agg = df[["r_direct", "r_direct_pos", "r_direct_neg", "r_indirect",
              "r_latent", "r_noise", "ratio_dir_indir", "ratio_dir_latent"]].agg(
        ["mean", "std"]).T
    print("\n" + "=" * 78)
    print("  RESUMEN AGREGADO (5 réplicas, media ± sd)")
    print("=" * 78)
    for name, row in agg.iterrows():
        print(f"  {name:<18}: {row['mean']:.4f} ± {row['std']:.4f}")

    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PILOTO")
    print("=" * 78)
    for k, v in [("1 prevalencia +-0.05 (vinculante)", c1),
                 ("2 direct>=2x indirect/latent POR REPLICA (solo orientativa)", c2),
                 ("3 indirect/latent no nulos (vinculante)", c3),
                 ("4 noise compatible con independencia (vinculante)", c4)]:
        print(f"  Comprobación {k:<58}: {'PASA' if v else 'NO PASA'}")
    allok = c1 and c3 and c4          # la 2 NO condiciona (Opción A -> se decide en Fase 2)
    print("-" * 78)
    if allok:
        print("  Comprobaciones vinculantes (1, 3, 4): PASAN.")
        print("  La separación direct/indirect se juzga sobre el promedio conjunto de")
        print("  la Fase 2 (Opción A). Lanzar:  python simulation_ground_truth.py --phase 2")
    else:
        print("  ALGO NO CUADRA en una comprobación vinculante. No ajusto parámetros por")
        print("  mi cuenta: repórtalo y lo revisamos (magnitud de w_direct/w_chain/w_latent).")
    print("=" * 78)
    print("\n>>> PARADA OBLIGATORIA: la Fase 2 NO se ejecuta sin tu aprobación. <<<")
    return allok


# =====================================================================
# FASE 2 — EVALUACIÓN COMPLETA  (sólo tras aprobación del piloto)
# =====================================================================
def _upper_pairs(n):
    ii, jj = np.triu_indices(n, k=1)
    return list(zip(ii.tolist(), jj.tolist()))


def _auc_pr(y_true, score):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, score))


def _fpr_in_category(score_vec, pair_index, cat_pairs, all_pairs, top_frac=0.30):
    """Proporción de pares de la categoría que caen en el top `top_frac` del método."""
    n_top = max(1, int(round(len(all_pairs) * top_frac)))
    order = np.argsort(score_vec)[::-1]
    top_set = {all_pairs[k] for k in order[:n_top]}
    hits = sum(1 for pr in cat_pairs if pr in top_set)
    return hits / len(cat_pairs) if cat_pairs else np.nan


def _method_scores(bin_df):
    """Devuelve dict método -> matriz (n x n) de scores con signo (sin filtrar)."""
    from inference_functions import (
        bernoulli_vae_network,
        bootstrap_vae_network,
        cooccurrence_phi_fisher_rows,
        neighborhood_selection_logistic,
    )
    warnings.filterwarnings("ignore")
    n = bin_df.shape[0]
    scores = {}
    notes = {}

    vae_params = load_vae_params()

    # --- VAE (un disparo) ---
    try:
        adj_vae, _, _ = bernoulli_vae_network(
            bin_df, normalize_adj=False, verbose=False, seed=0, **vae_params
        )
        scores["vae"] = np.asarray(adj_vae.values, dtype=float)
    except Exception as e:  # pragma: no cover
        notes["vae"] = f"ERROR: {e}"
        scores["vae"] = np.full((n, n), np.nan)

    # --- Bootstrap Mean (bagging n_bootstrap=30) ---
    try:
        consensus, _, _, _ = bootstrap_vae_network(
            bin_df, n_bootstrap=30, sample_fraction=0.6, vae_params=vae_params,
            normalize_for_stats=False, device="cpu", seed=0, verbose=False,
        )
        scores["bootstrap_mean"] = np.asarray(consensus.values, dtype=float)
    except Exception as e:  # pragma: no cover
        notes["bootstrap_mean"] = f"ERROR: {e}"
        scores["bootstrap_mean"] = np.full((n, n), np.nan)

    # --- Phi crudo (sin filtrar por threshold/FDR) ---
    try:
        _, phi_mat, _ = cooccurrence_phi_fisher_rows(bin_df, normalize_adj=False)
        scores["phi_raw"] = np.asarray(phi_mat, dtype=float)
    except Exception as e:  # pragma: no cover
        notes["phi_raw"] = f"ERROR: {e}"
        scores["phi_raw"] = np.full((n, n), np.nan)

    # --- L1 con signo (average_signed) ---
    try:
        adj_l1, _ = neighborhood_selection_logistic(
            bin_df, normalize_adj=False, symm_rule="average_signed",
        )
        scores["l1_signed"] = np.asarray(adj_l1.values, dtype=float)
    except Exception as e:  # pragma: no cover
        notes["l1_signed"] = f"ERROR: {e}"
        scores["l1_signed"] = np.full((n, n), np.nan)

    return scores, notes


def run_phase2():
    real = load_real_matrix()
    print("=" * 78)
    print("  FASE 2 — EVALUACIÓN COMPLETA CON VERDAD CONOCIDA")
    print("=" * 78)
    print(f"  Seeds                  : {PHASE2_SEEDS[0]}..{PHASE2_SEEDS[-1]} "
          f"({len(PHASE2_SEEDS)} réplicas)")
    print(f"  Parámetros simulación  : {SIM_KWARGS}")
    print(f"  bootstrap_mean         : bagging n_bootstrap=30 (declarado así por eficiencia)")
    print("-" * 78)

    methods = ["vae", "bootstrap_mean", "phi_raw", "l1_signed"]
    records = []
    all_notes = []
    # correlaciones marginales empíricas de la SIMULACIÓN, acumuladas sobre TODAS
    # las réplicas para juzgar la separación direct/indirect en el promedio
    # conjunto (Opción A aprobada), no por réplica.
    pooled_corr = {"direct": [], "indirect": [], "latent": [], "noise": []}

    for rep, seed in enumerate(PHASE2_SEEDS, 1):
        bin_df, gt = generate_synthetic_ground_truth(
            seed=seed, real_matrix=real, **SIM_KWARGS
        )
        cc = category_corrs(bin_df, gt)
        for key in pooled_corr:
            pooled_corr[key].append(np.abs(cc[key]))
        n = bin_df.shape[0]
        all_pairs = _upper_pairs(n)
        pair_index = {pr: k for k, pr in enumerate(all_pairs)}

        pos = {(i, j) for i, j, _ in gt["direct"]}
        indir = [(i, j) for i, j in gt["indirect"]]
        lat = [(i, j) for i, j in gt["latent"]]
        neg = set(indir) | set(lat) | set(gt["noise"])
        y_true = np.array([1 if pr in pos else (0 if pr in neg else -1)
                           for pr in all_pairs])
        keep = y_true >= 0
        y_eval = y_true[keep]
        pairs_eval = [pr for pr, k in zip(all_pairs, keep) if k]

        scores, notes = _method_scores(bin_df)
        for m, msg in notes.items():
            all_notes.append(f"seed {seed} / {m}: {msg}")

        for m in methods:
            S = scores[m]
            if not np.all(np.isfinite(S[np.triu_indices(n, 1)])):
                records.append(dict(seed=seed, method=m, auc_pr=np.nan,
                                    fpr_indirect=np.nan, fpr_latent=np.nan,
                                    failed=True))
                continue
            svec_all = np.array([abs(S[i, j]) for i, j in all_pairs])
            svec_eval = np.array([abs(S[i, j]) for i, j in pairs_eval])
            auc = _auc_pr(y_eval, svec_eval)
            fpr_i = _fpr_in_category(svec_all, pair_index, indir, all_pairs, 0.30)
            fpr_l = _fpr_in_category(svec_all, pair_index, lat, all_pairs, 0.30)
            records.append(dict(seed=seed, method=m, auc_pr=auc,
                                fpr_indirect=fpr_i, fpr_latent=fpr_l, failed=False))
        print(f"  réplica {rep:>2}/{len(PHASE2_SEEDS)} (seed {seed}) hecha")

    res = pd.DataFrame.from_records(records)
    res.to_csv(os.path.join(OUT_DIR, "per_replicate_results.csv"), index=False)

    summary = (res[~res["failed"]]
               .groupby("method")[["auc_pr", "fpr_indirect", "fpr_latent"]]
               .agg(["mean", "std", "count"]))
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"))

    print("\n" + "=" * 78)
    print("  RESUMEN FASE 2  (media ± sd sobre réplicas)")
    print("=" * 78)
    for m in methods:
        sub = res[(res["method"] == m) & (~res["failed"])]
        nfail = int(((res["method"] == m) & (res["failed"])).sum())
        if sub.empty:
            print(f"  {m:<16}: sin réplicas válidas ({nfail} fallos)")
            continue
        print(f"  {m:<16}: AUC-PR = {sub['auc_pr'].mean():.3f} ± {sub['auc_pr'].std():.3f}"
              f"  | FPR indirect = {sub['fpr_indirect'].mean():.3f} ± {sub['fpr_indirect'].std():.3f}"
              f"  | FPR latent = {sub['fpr_latent'].mean():.3f} ± {sub['fpr_latent'].std():.3f}"
              f"  | n={len(sub)} fallos={nfail}")

    print("\n  Nota: AUC-PR base (positivos/total evaluados) ~ "
          f"{len(pos)/len(pairs_eval):.3f}")

    # ---- Separación direct/indirect en el PROMEDIO CONJUNTO (Opción A) ----
    pooled = {k: np.concatenate(v) for k, v in pooled_corr.items()}
    m_dir = float(pooled["direct"].mean())
    m_ind = float(pooled["indirect"].mean())
    m_lat = float(pooled["latent"].mean())
    m_noi = float(pooled["noise"].mean())
    ratio_di = m_dir / m_ind if m_ind > 0 else np.inf
    ratio_dl = m_dir / m_lat if m_lat > 0 else np.inf
    pd.DataFrame([dict(
        n_replicas=len(PHASE2_SEEDS),
        r_direct=m_dir, r_indirect=m_ind, r_latent=m_lat, r_noise=m_noi,
        n_direct=pooled["direct"].size, n_indirect=pooled["indirect"].size,
        n_latent=pooled["latent"].size, n_noise=pooled["noise"].size,
        ratio_dir_indir=ratio_di, ratio_dir_latent=ratio_dl,
        criterio_min=PHASE2_SEP_RATIO_MIN,
        null_abs_r=NULL_ABS_R,
    )]).to_csv(os.path.join(OUT_DIR, "phase2_separation_pooled.csv"), index=False)

    print("\n" + "=" * 78)
    print("  SEPARACIÓN direct/indirect — PROMEDIO CONJUNTO DE LAS RÉPLICAS (Opción A)")
    print("=" * 78)
    print(f"  |r| direct   (pooled, n={pooled['direct'].size:>4}) : {m_dir:.4f}")
    print(f"  |r| indirect (pooled, n={pooled['indirect'].size:>4}) : {m_ind:.4f}")
    print(f"  |r| latent   (pooled, n={pooled['latent'].size:>4}) : {m_lat:.4f}")
    print(f"  |r| noise    (pooled, n={pooled['noise'].size:>4}) : {m_noi:.4f}  "
          f"(nulo analítico {NULL_ABS_R:.4f}, ratio {m_noi / NULL_ABS_R:.2f}x)")
    print(f"  ratio direct/indirect = {ratio_di:.2f}x   "
          f"(criterio prefijado >= {PHASE2_SEP_RATIO_MIN:g}x  -> "
          f"{'CUMPLE' if ratio_di >= PHASE2_SEP_RATIO_MIN else 'NO CUMPLE'})")
    print(f"  ratio direct/latent   = {ratio_dl:.2f}x   "
          f"(criterio prefijado >= {PHASE2_SEP_RATIO_MIN:g}x  -> "
          f"{'CUMPLE' if ratio_dl >= PHASE2_SEP_RATIO_MIN else 'NO CUMPLE'})")

    print("\n" + "-" * 78)
    if all_notes:
        print("  ADVERTENCIAS / FALLOS DURANTE LA SIMULACIÓN:")
        for ln in all_notes:
            print(f"    - {ln}")
    else:
        print("  Sin advertencias de no-convergencia ni fallos registrados.")
    print("=" * 78)
    print(f"\n  CSVs -> {OUT_DIR}")


# =====================================================================
# FASE 2b — AÑADIR bootstrap_stat A LA EVALUACIÓN YA HECHA
# ---------------------------------------------------------------------
# bootstrap_stat = bootstrap_vae_statistical_network: del ajuste sobre el
# disparo único COMPLETO retiene sólo los arcos cuyo valor supera un umbral
# de SOPORTE bootstrap por posición (percentil de la distribución bootstrap
# de esa arista), en vez de promediar las réplicas como bootstrap_mean.
#
# Parámetros para esta validación (decididos ANTES de ver resultados):
#   - n_bootstrap    = 30   (igual que bootstrap_mean aquí, no 75)
#   - sample_fraction= 0.6  (igual que bootstrap_mean / bootstrap_stat del pipeline)
#   - umbral soporte = percentil 95 de los valores bootstrap por posición
#   - MISMAS 35 réplicas (seeds 100..134): generate_synthetic_ground_truth es
#     determinista por seed -> los datos regenerados son idénticos a los de la
#     Fase 2, la comparación es pareada.
#
# La función bootstrap_vae_statistical_network NO se modifica. Su umbral
# interno (`vae_params['percentile_threshold']`, por defecto 80) se ignora:
# se reconstruye la red estadística al percentil 95 a partir de las matrices
# que la propia función devuelve (S_full y S_boot), sin re-entrenar nada.
# =====================================================================
BOOTSTRAP_STAT_SUPPORT_PCTL = 95
BOOTSTRAP_STAT_NBOOT = 30
BOOTSTRAP_STAT_FRAC = 0.6


def _eval_score_matrix(S, gt, n):
    """(auc_pr, fpr_indirect, fpr_latent) con EXACTAMENTE el mismo protocolo
    que run_phase2: positivos = pares 'direct', negativos = indirect+latent+noise,
    score = |S_ij|, FPR = fracción de la categoría en el top 30 %."""
    all_pairs = _upper_pairs(n)
    pair_index = {pr: k for k, pr in enumerate(all_pairs)}
    pos = {(i, j) for i, j, _ in gt["direct"]}
    indir = [(i, j) for i, j in gt["indirect"]]
    lat = [(i, j) for i, j in gt["latent"]]
    neg = set(indir) | set(lat) | set(gt["noise"])
    y_true = np.array([1 if pr in pos else (0 if pr in neg else -1) for pr in all_pairs])
    keep = y_true >= 0
    y_eval = y_true[keep]
    pairs_eval = [pr for pr, k in zip(all_pairs, keep) if k]

    if not np.all(np.isfinite(S[np.triu_indices(n, 1)])):
        return np.nan, np.nan, np.nan, True
    svec_all = np.array([abs(S[i, j]) for i, j in all_pairs])
    svec_eval = np.array([abs(S[i, j]) for i, j in pairs_eval])
    auc = _auc_pr(y_eval, svec_eval)
    fpr_i = _fpr_in_category(svec_all, pair_index, indir, all_pairs, 0.30)
    fpr_l = _fpr_in_category(svec_all, pair_index, lat, all_pairs, 0.30)
    return auc, fpr_i, fpr_l, False


# =====================================================================
# VAE hyperparameter selection by ground-truth recovery accuracy. The
# seed-to-seed STABILITY search in pipeline_main.py Block 3 is diagnostic only
# and does not determine the operative hyperparameters (see the note at the
# top of that block).
#
# Why this belongs here and not in pipeline_main.py: this reuses the same
# synthetic-ground-truth machinery (generate_synthetic_ground_truth,
# _eval_score_matrix) already defined in this file for the method
# comparison, and must run on TUNING_SEEDS (200-229), a block disjoint from
# both the pilot (0-4) and the final evaluation block (100-134) that reports
# Table 1 / the manuscript's method comparison -- using the same replicates
# for tuning and reporting would be data leakage.
#
# SELECTION CRITERION -- composite score. Two simpler criteria were rejected,
# for the reasons below:
#
#   REJECTED #1 -- pure mean_fpr_latent (lower is better), tie-broken by
#   mean_auc_pr. This is what the manuscript declares as its primary metric
#   for the 6/7-method comparison (PRIMARY_FPR_LATENT_PAIRS), so it looked
#   like the natural choice for hyperparameter selection too. In practice it
#   systematically favoured degenerate LOW-CAPACITY configs (latent_dim=8,
#   hidden_dim=8, high beta_kl): those configs "win" on fpr_latent simply by
#   detecting almost nothing at all (mean_auc_pr collapses to ~0.17-0.27,
#   vs. ~0.65-0.71 for reasonable-capacity configs) -- not by being better at
#   avoiding false latent-driven edges. A criterion that rewards a model for
#   refusing to make any calls is not a criterion the manuscript can defend.
#
#   REJECTED #2 -- filter configs to mean_auc_pr >= some fixed floor (e.g.
#   0.5), then pick the lowest mean_fpr_latent among survivors. This avoids
#   the degenerate-model failure mode of #1, but the floor itself is
#   arbitrary: sorting all 48 configs by mean_auc_pr, the gap that straddles
#   0.5 (0.5488 -> 0.5120) is NOT the largest gap in the distribution -- a
#   larger gap (0.4755 -> 0.4326) sits further down, and a comparably-sized
#   gap sits at the very top (0.7101 -> 0.6792). There is no single
#   objectively "clean" separation in this data, so any fixed floor is a
#   judgment call dressed up as an objective threshold.
#
#   SELECTED CRITERION -- composite score, a harmonic mean of min-max-normalized
#   mean_auc_pr and mean specificity against latent-driven false edges
#   (mean_spec_latent = 1 - mean_fpr_latent), computed across the 48
#   configs:
#     norm_x = (x - min_48(x)) / (max_48(x) - min_48(x))   for x in {auc_pr, spec_latent}
#     score  = 2 * norm_auc_pr * norm_spec_latent / (norm_auc_pr + norm_spec_latent)
#            (score = 0 if norm_auc_pr + norm_spec_latent == 0, avoiding /0)
#   Min-max normalization within the 48-config set is REQUIRED: mean_auc_pr
#   ranges over ~0.54 (0.17-0.71) while mean_fpr_latent ranges over only
#   ~0.19 (0.32-0.53) -- an unnormalized combination would be dominated by
#   auc_pr almost exclusively, defeating the purpose of including fpr_latent
#   at all. The harmonic mean (vs. an arithmetic mean) penalizes configs that
#   are lopsided on either axis, which is exactly what rules out both
#   degenerate low-capacity configs (weak on norm_auc_pr) and configs that
#   detect too indiscriminately (weak on norm_spec_latent).
#
# COMPLEMENTARY criterion, reported but NOT part of the composite score:
# held-out ELBO / log-likelihood via k-fold CV on the REAL 63x150 matrix
# (evaluate_vae_held_out_likelihood, point 4) -- goodness of fit to the
# OBSERVED data, which the ground-truth criterion cannot measure (the real
# data has no known ground truth). Whether its own single-criterion winner
# agrees with the composite-score winner is reported explicitly, never
# silently resolved.
# =====================================================================
HP_SEARCH_SPACE = {
    "latent_dim": [8, 16, 32],
    "hidden_dim": [8, 16, 32, 64],
    "beta_kl": [0.01, 0.05, 0.1, 0.5],
}


def _hp_grid():
    from itertools import product
    keys = list(HP_SEARCH_SPACE.keys())
    values = list(HP_SEARCH_SPACE.values())
    return [dict(zip(keys, vals)) for vals in product(*values)]


def _write_selected_config(score_winner_params, score_winner_row, ho_winner_row,
                          hardcoded_winner, agree_with_held_out):
    from vae_config_loader import load_vae_config, DEFAULT_CONFIG_PATH
    cfg = load_vae_config()
    cfg["selected_params"] = score_winner_params
    cfg["_selection_method"] = "ground_truth_composite_harmonic_mean_normalized_aucpr_specificity_latent"
    note = (
        f"Selected by simulation_ground_truth.py "
        f"run_hyperparameter_search_ground_truth() on tuning seeds "
        f"{TUNING_SEEDS[0]}-{TUNING_SEEDS[-1]} ({len(TUNING_SEEDS)} replicas, disjoint "
        f"from pilot 0-4 and final evaluation 100-134). "
        f"FINAL criterion (see code comment above run_hyperparameter_search_ground_truth "
        f"for the two discarded alternatives and why): harmonic mean of "
        f"min-max-normalized mean_auc_pr and mean_spec_latent (=1-mean_fpr_latent) "
        f"across the 48 configs. "
        f"Composite-score winner: {score_winner_params} "
        f"(score={score_winner_row['score']:.4f}, mean_auc_pr={score_winner_row['mean_auc_pr']:.4f}, "
        f"mean_fpr_latent={score_winner_row['mean_fpr_latent']:.4f}). "
        f"Held-out log-likelihood winner (complementary, real-data k-fold CV, NOT part of "
        f"the composite score): latent_dim={int(ho_winner_row['latent_dim'])}, "
        f"hidden_dim={int(ho_winner_row['hidden_dim'])}, "
        f"beta_kl={float(ho_winner_row['beta_kl'])} "
        f"(mean_log_likelihood_held_out={ho_winner_row['mean_log_likelihood_held_out']:.4f}). "
        f"Composite-score winner matches held-out log-likelihood winner: {agree_with_held_out}. "
        f"vae_config.json's selected_params before this run "
        f"(pipeline_main.py Block 3's stability-based winner, diagnostic-only, "
        f"selects by seed-to-seed stability rather than ground-truth accuracy): "
        f"{hardcoded_winner}. "
        f"Composite-score winner matches old hardcoded config: "
        f"{score_winner_params == hardcoded_winner}."
    )
    cfg["_selection_note"] = note
    with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    print(f"\n  -> updated: {DEFAULT_CONFIG_PATH}")


def run_hyperparameter_search_ground_truth(k_folds_held_out=5):
    from inference_functions import bernoulli_vae_network, evaluate_vae_held_out_likelihood
    from vae_config_loader import load_vae_config

    warnings.filterwarnings("ignore")
    cfg = load_vae_config()
    base_params = dict(cfg["base_params"])
    hardcoded_winner = dict(cfg["selected_params"])

    real = load_real_matrix()
    grid = _hp_grid()

    print("=" * 78)
    print("  POINT 1 -- VAE HYPERPARAMETER SEARCH ON SIMULATED GROUND TRUTH")
    print("=" * 78)
    print(f"  Tuning seeds     : {TUNING_SEEDS[0]}..{TUNING_SEEDS[-1]} ({len(TUNING_SEEDS)} replicas)")
    print(f"  Simulation kwargs: {SIM_KWARGS}")
    print(f"  Configurations   : {len(grid)}")
    print("-" * 78)

    # Pre-generate the 30 tuning replicates ONCE, reused for all 48 configs
    # (generate_synthetic_ground_truth is deterministic per seed).
    tuning_data = [
        (seed,) + generate_synthetic_ground_truth(seed=seed, real_matrix=real, **SIM_KWARGS)
        for seed in TUNING_SEEDS
    ]

    gt_rows = []
    for config_idx, hp in enumerate(grid, start=1):
        vae_params = {**base_params, **hp}
        print(f"  config {config_idx:02d}/{len(grid)}: {hp}")
        aucs, fpr_is, fpr_ls = [], [], []
        for seed, bin_df, gt in tuning_data:
            n = bin_df.shape[0]
            try:
                adj_vae, _, _ = bernoulli_vae_network(
                    bin_df, normalize_adj=False, verbose=False, seed=0, **vae_params
                )
                S = np.asarray(adj_vae.values, dtype=float)
                auc, fpr_i, fpr_l, failed = _eval_score_matrix(S, gt, n)
            except Exception:
                auc, fpr_i, fpr_l, failed = np.nan, np.nan, np.nan, True
            if not failed:
                aucs.append(auc)
                fpr_is.append(fpr_i)
                fpr_ls.append(fpr_l)
        gt_rows.append(dict(
            config_id=config_idx, **hp,
            n_replicas_ok=len(aucs),
            mean_auc_pr=float(np.mean(aucs)) if aucs else np.nan,
            std_auc_pr=float(np.std(aucs)) if aucs else np.nan,
            mean_fpr_indirect=float(np.mean(fpr_is)) if fpr_is else np.nan,
            std_fpr_indirect=float(np.std(fpr_is)) if fpr_is else np.nan,
            mean_fpr_latent=float(np.mean(fpr_ls)) if fpr_ls else np.nan,
            std_fpr_latent=float(np.std(fpr_ls)) if fpr_ls else np.nan,
        ))
        print(f"    mean_fpr_latent={gt_rows[-1]['mean_fpr_latent']:.4f}  "
              f"mean_auc_pr={gt_rows[-1]['mean_auc_pr']:.4f}")

    gt_df = pd.DataFrame(gt_rows)

    # ---- complementary criterion (point 4): held-out log-likelihood, real data ----
    print("\n" + "-" * 78)
    print(f"  COMPLEMENTARY CRITERION -- held-out ELBO / log-likelihood "
          f"(real data, k={k_folds_held_out}-fold CV)")
    print("-" * 78)
    lineage_csv = os.path.join(ROOT, "results", "phylogenetic_sensitivity", "lineage_clusters.csv")
    lineage_csv = lineage_csv if os.path.exists(lineage_csv) else None
    ho_rows, ho_summary = [], []
    for config_idx, hp in enumerate(grid, start=1):
        vae_params = {**base_params, **hp}
        ho = evaluate_vae_held_out_likelihood(
            real, vae_params, k_folds=k_folds_held_out, seed=42,
            lineage_csv=lineage_csv, verbose=False,
        )
        for _, row in ho.iterrows():
            # `stratified`/`stratify_note` (returned per-row by
            # evaluate_vae_held_out_likelihood) record whether the folds were
            # lineage-stratified or fell back to random KFold, so
            # held_out_evaluation.csv shows it. They depend only on the
            # fold-splitting step (real matrix, lineage_clusters.csv, k_folds,
            # seed), so they are identical for every config.
            ho_rows.append(dict(config_id=config_idx, **hp, fold=int(row["fold"]),
                                elbo_held_out=row["elbo_held_out"],
                                log_likelihood_held_out=row["log_likelihood_held_out"],
                                stratified=bool(row["stratified"]),
                                stratify_note=row["stratify_note"]))
        ho_summary.append(dict(
            config_id=config_idx, **hp,
            mean_log_likelihood_held_out=float(ho["log_likelihood_held_out"].mean()),
            mean_elbo_held_out=float(ho["elbo_held_out"].mean()),
        ))
        print(f"  config {config_idx:02d}/{len(grid)}: "
              f"mean logL_held_out={ho_summary[-1]['mean_log_likelihood_held_out']:.4f}")

    ho_df = pd.DataFrame(ho_rows)
    ho_summary_df = pd.DataFrame(ho_summary)
    ho_df.to_csv(os.path.join(OUT_DIR, "held_out_evaluation.csv"), index=False)

    merge_keys = ["config_id", "latent_dim", "hidden_dim", "beta_kl"]
    gt_df = gt_df.merge(ho_summary_df, on=merge_keys, how="left")
    gt_df.to_csv(os.path.join(OUT_DIR, "TableS2.csv"), index=False)

    return _select_winner_and_write_config(gt_df, hardcoded_winner)


def _select_winner_and_write_config(gt_df, hardcoded_winner):
    """
    Apply the FINAL composite-score selection criterion to an already-computed
    gt_df (columns: config_id, latent_dim, hidden_dim, beta_kl, mean_auc_pr,
    mean_fpr_latent, mean_log_likelihood_held_out, ...), write
    hyperparameter_search_ground_truth_composite.csv, and update
    src/vae_config.json. Split out from run_hyperparameter_search_ground_truth()
    so the (expensive, ~1.5-2h) 48-config x 30-replica + 48-config x k-fold
    computation never needs to be re-run just to re-derive the winner from
    already-computed summary statistics (e.g. after changing the scoring
    formula itself).
    """
    # ---- Composite harmonic-mean score (see the long
    # comment above run_hyperparameter_search_ground_truth for why the two
    # simpler alternatives were rejected) ----
    gt_df = gt_df.copy()
    gt_df["mean_spec_latent"] = 1.0 - gt_df["mean_fpr_latent"]
    auc_min, auc_max = gt_df["mean_auc_pr"].min(), gt_df["mean_auc_pr"].max()
    spec_min, spec_max = gt_df["mean_spec_latent"].min(), gt_df["mean_spec_latent"].max()
    gt_df["norm_auc_pr"] = (gt_df["mean_auc_pr"] - auc_min) / (auc_max - auc_min)
    gt_df["norm_spec_latent"] = (gt_df["mean_spec_latent"] - spec_min) / (spec_max - spec_min)
    denom = gt_df["norm_auc_pr"] + gt_df["norm_spec_latent"]
    gt_df["score"] = np.where(
        denom > 0,
        2 * gt_df["norm_auc_pr"] * gt_df["norm_spec_latent"] / denom.replace(0, np.nan),
        0.0,
    )
    gt_df["score"] = gt_df["score"].fillna(0.0)

    composite_cols = ["config_id", "latent_dim", "hidden_dim", "beta_kl",
                      "mean_auc_pr", "mean_fpr_latent", "norm_auc_pr",
                      "norm_spec_latent", "score", "mean_log_likelihood_held_out"]
    composite_df = gt_df.sort_values("score", ascending=False)[composite_cols]
    composite_df.to_csv(
        os.path.join(OUT_DIR, "hyperparameter_search_ground_truth_composite.csv"),
        index=False,
    )

    score_winner = composite_df.iloc[0]
    score_winner_params = dict(latent_dim=int(score_winner["latent_dim"]),
                               hidden_dim=int(score_winner["hidden_dim"]),
                               beta_kl=float(score_winner["beta_kl"]))

    ho_sorted = gt_df.sort_values("mean_log_likelihood_held_out", ascending=False)
    ho_winner = ho_sorted.iloc[0]
    ho_winner_params = dict(latent_dim=int(ho_winner["latent_dim"]),
                            hidden_dim=int(ho_winner["hidden_dim"]),
                            beta_kl=float(ho_winner["beta_kl"]))

    agree_with_held_out = score_winner_params == ho_winner_params
    matches_hardcoded = score_winner_params == hardcoded_winner

    print("\n" + "=" * 78)
    print("  RESULT -- FINAL composite-score criterion")
    print("=" * 78)
    print("  Composite-score winner (harmonic mean of normalized auc_pr and spec_latent):")
    print(f"    {score_winner_params}  score={score_winner['score']:.4f}  "
          f"mean_auc_pr={score_winner['mean_auc_pr']:.4f}  "
          f"mean_fpr_latent={score_winner['mean_fpr_latent']:.4f}")
    print("  Held-out log-likelihood winner (complementary, NOT part of the score):")
    print(f"    {ho_winner_params}  "
          f"mean_logL_held_out={ho_winner['mean_log_likelihood_held_out']:.4f}")
    print(f"  Composite-score winner matches held-out LL winner : "
          f"{'YES' if agree_with_held_out else 'NO'}")
    print(f"  Previously hardcoded (build_final_network.py, old stability search): "
          f"{hardcoded_winner}")
    print(f"  Composite-score winner matches old hardcoded config : "
          f"{'YES' if matches_hardcoded else 'NO'}")

    _write_selected_config(score_winner_params, score_winner, ho_winner, hardcoded_winner,
                          agree_with_held_out=agree_with_held_out)

    return gt_df, composite_df, score_winner_params, ho_winner_params, agree_with_held_out


def _bootstrap_stat_score(bin_df):
    """Matriz de score (n x n) de bootstrap_stat al percentil 95 de soporte."""
    from inference_functions import bootstrap_vae_statistical_network
    warnings.filterwarnings("ignore")
    n = bin_df.shape[0]
    vae_params = load_vae_params()
    try:
        _, S_full_df, S_boot = bootstrap_vae_statistical_network(
            bin_df, vae_params=vae_params,
            n_bootstrap=BOOTSTRAP_STAT_NBOOT, sample_fraction=BOOTSTRAP_STAT_FRAC,
            normalize_for_stats=False, device="cpu", seed=0, verbose=False,
        )
        S_full = np.asarray(S_full_df.values, dtype=float)
        S_boot = np.asarray(S_boot, dtype=float)              # (B, n, n)
        thr = np.percentile(S_boot, BOOTSTRAP_STAT_SUPPORT_PCTL, axis=0)   # soporte por posición
        mask = S_full >= thr                                  # arco retenido si supera su propio soporte
        stat_adj = np.where(mask, S_full, 0.0)
        np.fill_diagonal(stat_adj, 0.0)
        return stat_adj, None
    except Exception as e:  # pragma: no cover
        return np.full((n, n), np.nan), f"ERROR: {e}"


def _paired_wilcoxon(a, b):
    """Wilcoxon pareado de dos colas sobre vectores emparejados a, b."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    diff = a - b
    out = dict(n=int(ok.sum()), median_a=float(np.median(a)), median_b=float(np.median(b)),
               median_diff=float(np.median(diff)),
               n_a_gt_b=int(np.sum(diff > 0)), n_b_gt_a=int(np.sum(diff < 0)),
               n_ties=int(np.sum(diff == 0)))
    if np.allclose(diff, 0):
        out.update(stat=np.nan, p=1.0, note="todas las diferencias nulas")
        return out
    try:
        st, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        st, p = wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
    out.update(stat=float(st), p=float(p), note="")
    return out


def run_phase2b_bootstrap_stat():
    per_rep_csv = os.path.join(OUT_DIR, "per_replicate_results.csv")
    summ_csv = os.path.join(OUT_DIR, "summary.csv")
    if not os.path.exists(per_rep_csv):
        raise FileNotFoundError(f"Falta {per_rep_csv}: corre antes la Fase 2.")

    prev = pd.read_csv(per_rep_csv, float_precision="round_trip")
    if "bootstrap_stat" in set(prev["method"]):
        print("bootstrap_stat ya está en phase2_per_replicate.csv. Nada que hacer.")
        return
    # copia de seguridad de los 4 métodos originales
    prev.to_csv(os.path.join(OUT_DIR, "phase2_per_replicate_4methods_backup.csv"), index=False)
    if os.path.exists(summ_csv):
        pd.read_csv(summ_csv, float_precision="round_trip").to_csv(
            os.path.join(OUT_DIR, "phase2_summary_4methods_backup.csv"), index=False)

    real = load_real_matrix()
    print("=" * 78)
    print("  FASE 2b — AÑADIR bootstrap_stat (bootstrap_vae_statistical_network)")
    print("=" * 78)
    print(f"  Seeds                  : {PHASE2_SEEDS[0]}..{PHASE2_SEEDS[-1]} "
          f"({len(PHASE2_SEEDS)} réplicas, MISMAS que la Fase 2)")
    print(f"  n_bootstrap            : {BOOTSTRAP_STAT_NBOOT}")
    print(f"  sample_fraction        : {BOOTSTRAP_STAT_FRAC}")
    print(f"  umbral de soporte      : percentil {BOOTSTRAP_STAT_SUPPORT_PCTL} "
          f"de los valores bootstrap por posición (fijado a priori)")
    print("-" * 78)

    new_rows = []
    notes = []
    for rep, seed in enumerate(PHASE2_SEEDS, 1):
        bin_df, gt = generate_synthetic_ground_truth(seed=seed, real_matrix=real, **SIM_KWARGS)
        n = bin_df.shape[0]
        S, err = _bootstrap_stat_score(bin_df)
        if err:
            notes.append(f"seed {seed} / bootstrap_stat: {err}")
        auc, fpr_i, fpr_l, failed = _eval_score_matrix(S, gt, n)
        new_rows.append(dict(seed=seed, method="bootstrap_stat", auc_pr=auc,
                             fpr_indirect=fpr_i, fpr_latent=fpr_l, failed=failed))
        print(f"  réplica {rep:>2}/{len(PHASE2_SEEDS)} (seed {seed}) hecha")

    new_df = pd.DataFrame(new_rows)[list(prev.columns)]
    combined = pd.concat([prev, new_df], ignore_index=True)

    # comprobación (ANTES de escribir): los 4 métodos originales NO cambian
    check = combined[combined["method"] != "bootstrap_stat"].reset_index(drop=True)
    assert check.equals(prev.reset_index(drop=True)), "Se han alterado filas de los métodos originales"
    combined.to_csv(per_rep_csv, index=False)

    summary = (combined[~combined["failed"]]
               .groupby("method")[["auc_pr", "fpr_indirect", "fpr_latent"]]
               .agg(["mean", "std", "count"]))
    summary.to_csv(summ_csv)

    methods = ["vae", "bootstrap_mean", "bootstrap_stat", "phi_raw", "l1_signed"]
    print("\n" + "=" * 78)
    print("  RESUMEN FASE 2 — 5 MÉTODOS (media ± sd sobre 35 réplicas)")
    print("=" * 78)
    print(f"  {'método':<16} {'AUC-PR':>16} {'FPR indirect':>18} {'FPR latent':>18}")
    for m in methods:
        sub = combined[(combined["method"] == m) & (~combined["failed"])]
        nfail = int(((combined["method"] == m) & (combined["failed"])).sum())
        print(f"  {m:<16} {sub['auc_pr'].mean():>7.3f} ± {sub['auc_pr'].std():<6.3f}"
              f" {sub['fpr_indirect'].mean():>8.3f} ± {sub['fpr_indirect'].std():<6.3f}"
              f" {sub['fpr_latent'].mean():>8.3f} ± {sub['fpr_latent'].std():<6.3f}"
              f"   (n={len(sub)}, fallos={nfail})")
    print("\n  Nota: AUC-PR base ~ 0.028 (12 positivos / 435 pares).")

    # ---- Wilcoxon pareado ----------------------------------------
    wide = combined.pivot_table(index="seed", columns="method", values="auc_pr")
    wide_fl = combined.pivot_table(index="seed", columns="method", values="fpr_latent")

    print("\n" + "=" * 78)
    print("  WILCOXON PAREADO (dos colas, 35 réplicas emparejadas)")
    print("=" * 78)
    print("  NOTA: no se corrió Wilcoxon en la vuelta anterior; se ejecuta ahora")
    print("  para TODAS las comparaciones pedidas sobre las mismas 35 réplicas.")

    auc_pairs = [("vae", "bootstrap_mean"), ("vae", "phi_raw"), ("vae", "l1_signed"),
                 ("bootstrap_stat", "vae"), ("bootstrap_stat", "bootstrap_mean")]
    fl_pairs = [("bootstrap_stat", "vae"), ("bootstrap_stat", "bootstrap_mean")]

    def _report(tag, table, pairs):
        print(f"\n  --- {tag} ---")
        print(f"  {'A vs B':<34} {'med A':>8} {'med B':>8} {'medΔ(A-B)':>11} "
              f"{'A>B/B>A':>9} {'W':>9} {'p':>10}")
        rows = []
        for A, B in pairs:
            r = _paired_wilcoxon(table[A].values, table[B].values)
            rows.append(dict(metric=tag, A=A, B=B, **r))
            wtxt = "nan" if not np.isfinite(r["stat"]) else f"{r['stat']:.1f}"
            print(f"  {A+' vs '+B:<34} {r['median_a']:>8.3f} {r['median_b']:>8.3f} "
                  f"{r['median_diff']:>11.3f} {str(r['n_a_gt_b'])+'/'+str(r['n_b_gt_a']):>9} "
                  f"{wtxt:>9} {r['p']:>10.4g}")
        return rows

    wrows = []
    wrows += _report("AUC-PR", wide, auc_pairs)
    wrows += _report("fpr_latent", wide_fl, fl_pairs)
    pd.DataFrame(wrows).to_csv(os.path.join(OUT_DIR, "wilcoxon_tests.csv"), index=False)

    print("\n" + "-" * 78)
    if notes:
        print("  ADVERTENCIAS / FALLOS bootstrap_stat:")
        for ln in notes:
            print(f"    - {ln}")
    else:
        print("  bootstrap_stat: sin advertencias de no-convergencia ni fallos registrados.")
    print("=" * 78)
    print(f"\n  Actualizado -> {per_rep_csv}")
    print(f"  Actualizado -> {summ_csv}")
    print(f"  Nuevo       -> {os.path.join(OUT_DIR, 'phase2_wilcoxon.csv')}")
    print(f"  Backups 4-métodos -> phase2_*_4methods_backup.csv")


# =====================================================================
# FASE 2c — AÑADIR l1_logistic_bagged A LA EVALUACIÓN YA COMPLETADA
# ---------------------------------------------------------------------
# l1_logistic_bagged = bagged_l1_network: bagging de la regresión logística L1
# (neighborhood_selection_logistic con symm_rule='average_signed', signo
# conservado), promediando las réplicas elemento a elemento. Se añade para que
# la validación con verdad conocida sea coherente con la Tabla 1, que reporta
# L1 bageado.
#
# Parámetros para esta validación (coherencia interna de esta prueba):
#   - n_bootstrap    = 30   (igual que bootstrap_mean / bootstrap_stat aquí;
#                            NO el 50 del pipeline ni el 75 de la Tabla 1)
#   - sample_fraction= 0.6  (igual que el resto de métodos bootstrap aquí)
#   - seed           = 0    -> bagged_l1_network fija np.random.seed(seed+b) por
#                            réplica (mecanismo ya implementado; no se toca).
#   - MISMAS 35 réplicas sintéticas (seeds 100..134), regeneradas de forma
#     determinista -> comparación pareada con los otros 5 métodos.
# bagged_l1_network NO se modifica; sólo se la llama.
# =====================================================================
L1_BAGGED_NBOOT = 30
L1_BAGGED_FRAC = 0.6

# Wilcoxon: comparaciones ya existentes + las nuevas de esta fase.
WILCOXON_AUC_PAIRS = [
    ("vae", "bootstrap_mean"), ("vae", "phi_raw"), ("vae", "l1_signed"),
    ("bootstrap_stat", "vae"), ("bootstrap_stat", "bootstrap_mean"),
    ("l1_logistic_bagged", "vae"), ("l1_logistic_bagged", "l1_signed"),
    ("l1_logistic_bagged", "bootstrap_mean"),
]
WILCOXON_FL_PAIRS = [
    ("bootstrap_stat", "vae"), ("bootstrap_stat", "bootstrap_mean"),
    ("l1_logistic_bagged", "vae"), ("l1_logistic_bagged", "l1_signed"),
    ("l1_logistic_bagged", "bootstrap_mean"),
]


def _l1_bagged_score(bin_df):
    """Matriz de score (n x n) CON SIGNO de l1_logistic_bagged (bagged_l1_network)."""
    from inference_functions import bagged_l1_network
    warnings.filterwarnings("ignore")
    n = bin_df.shape[0]
    try:
        consensus_df, _ = bagged_l1_network(
            bin_df, n_bootstrap=L1_BAGGED_NBOOT, sample_fraction=L1_BAGGED_FRAC,
            seed=0, n_jobs=-2,
        )
        return np.asarray(consensus_df.values, dtype=float), None
    except Exception as e:  # pragma: no cover
        return np.full((n, n), np.nan), f"ERROR: {e}"


def _wilcoxon_report(tag, table, pairs):
    print(f"\n  --- {tag} ---")
    print(f"  {'A vs B':<38} {'med A':>8} {'med B':>8} {'medΔ(A-B)':>11} "
          f"{'A>B/B>A':>9} {'W':>9} {'p':>11}")
    rows = []
    for A, B in pairs:
        r = _paired_wilcoxon(table[A].values, table[B].values)
        rows.append(dict(metric=tag, A=A, B=B, **r))
        wtxt = "nan" if not np.isfinite(r["stat"]) else f"{r['stat']:.1f}"
        print(f"  {A + ' vs ' + B:<38} {r['median_a']:>8.3f} {r['median_b']:>8.3f} "
              f"{r['median_diff']:>11.3f} {str(r['n_a_gt_b']) + '/' + str(r['n_b_gt_a']):>9} "
              f"{wtxt:>9} {r['p']:>11.4g}")
    return rows


def run_phase2c_l1_bagged():
    per_rep_csv = os.path.join(OUT_DIR, "per_replicate_results.csv")
    summ_csv = os.path.join(OUT_DIR, "summary.csv")
    wilcox_csv = os.path.join(OUT_DIR, "wilcoxon_tests.csv")
    if not os.path.exists(per_rep_csv):
        raise FileNotFoundError(f"Falta {per_rep_csv}: corre antes la Fase 2 / 2b.")

    prev = pd.read_csv(per_rep_csv, float_precision="round_trip")
    have = set(prev["method"])
    NEW = "l1_logistic_bagged"
    missing = {"vae", "bootstrap_mean", "phi_raw", "l1_signed", "bootstrap_stat"} - have
    if missing:
        raise RuntimeError(f"Faltan métodos previos {missing}: corre antes la Fase 2 y 2b.")

    print("=" * 78)
    print("  FASE 2c — AÑADIR l1_logistic_bagged (bagged_l1_network)")
    print("=" * 78)

    already_done = NEW in have
    if already_done:
        # Reanudable: las 35 réplicas ya están en el CSV -> no se recomputan,
        # sólo se regeneran resumen + Wilcoxon (p.ej. si una vuelta previa
        # abortó después de escribir per_replicate pero antes del Wilcoxon).
        print(f"  {NEW} ya presente en phase2_per_replicate.csv ({int((prev['method'] == NEW).sum())} "
              f"réplicas): NO se recomputa; sólo se regeneran resumen y Wilcoxon.")
        combined = prev.copy()
        notes = []
    else:
        # backup de los 5 métodos ya presentes (sólo la primera vez)
        prev.to_csv(os.path.join(OUT_DIR, "phase2_per_replicate_5methods_backup.csv"), index=False)
        if os.path.exists(summ_csv):
            pd.read_csv(summ_csv, float_precision="round_trip").to_csv(
                os.path.join(OUT_DIR, "phase2_summary_5methods_backup.csv"), index=False)

        real = load_real_matrix()
        print(f"  Seeds                  : {PHASE2_SEEDS[0]}..{PHASE2_SEEDS[-1]} "
              f"({len(PHASE2_SEEDS)} réplicas, MISMAS que la Fase 2)")
        print(f"  n_bootstrap            : {L1_BAGGED_NBOOT}  (coherencia interna; no 50/75)")
        print(f"  sample_fraction        : {L1_BAGGED_FRAC}")
        print(f"  symm_rule              : average_signed (dentro de bagged_l1_network, sin tocar)")
        print("-" * 78)

        new_rows, notes = [], []
        for rep, seed in enumerate(PHASE2_SEEDS, 1):
            bin_df, gt = generate_synthetic_ground_truth(seed=seed, real_matrix=real, **SIM_KWARGS)
            n = bin_df.shape[0]
            S, err = _l1_bagged_score(bin_df)
            if err:
                notes.append(f"seed {seed} / {NEW}: {err}")
            auc, fpr_i, fpr_l, failed = _eval_score_matrix(S, gt, n)
            new_rows.append(dict(seed=seed, method=NEW, auc_pr=auc,
                                 fpr_indirect=fpr_i, fpr_latent=fpr_l, failed=failed))
            print(f"  réplica {rep:>2}/{len(PHASE2_SEEDS)} (seed {seed}) hecha")

        new_df = pd.DataFrame(new_rows)[list(prev.columns)]
        combined = pd.concat([prev, new_df], ignore_index=True)

        # comprobación (ANTES de escribir): las filas de los 5 métodos previos NO cambian
        check = combined[combined["method"] != NEW].reset_index(drop=True)
        assert check.equals(prev.reset_index(drop=True)), "Se han alterado filas de métodos previos"
        combined.to_csv(per_rep_csv, index=False)

    # backup del Wilcoxon previo (siempre, antes de reescribirlo)
    prev_wilcox = pd.read_csv(wilcox_csv, float_precision="round_trip") if os.path.exists(wilcox_csv) else None
    if prev_wilcox is not None:
        prev_wilcox.to_csv(os.path.join(OUT_DIR, "phase2_wilcoxon_prev_backup.csv"), index=False)

    summary = (combined[~combined["failed"]]
               .groupby("method")[["auc_pr", "fpr_indirect", "fpr_latent"]]
               .agg(["mean", "std", "count"]))
    summary.to_csv(summ_csv)

    methods = ["vae", "bootstrap_mean", "bootstrap_stat", "phi_raw",
               "l1_signed", "l1_logistic_bagged"]
    print("\n" + "=" * 78)
    print("  RESUMEN FASE 2 — 6 MÉTODOS (media ± sd sobre 35 réplicas)")
    print("=" * 78)
    print(f"  {'método':<20} {'AUC-PR':>16} {'FPR indirect':>18} {'FPR latent':>18}")
    for m in methods:
        sub = combined[(combined["method"] == m) & (~combined["failed"])]
        nfail = int(((combined["method"] == m) & (combined["failed"])).sum())
        print(f"  {m:<20} {sub['auc_pr'].mean():>7.3f} ± {sub['auc_pr'].std():<6.3f}"
              f" {sub['fpr_indirect'].mean():>8.3f} ± {sub['fpr_indirect'].std():<6.3f}"
              f" {sub['fpr_latent'].mean():>8.3f} ± {sub['fpr_latent'].std():<6.3f}"
              f"   (n={len(sub)}, fallos={nfail})")
    print("\n  Nota: AUC-PR base ~ 0.028 (12 positivos / 435 pares).")

    # ---- Wilcoxon pareado: comparaciones previas + nuevas -------------
    wide = combined.pivot_table(index="seed", columns="method", values="auc_pr")
    wide_fl = combined.pivot_table(index="seed", columns="method", values="fpr_latent")
    print("\n" + "=" * 78)
    print("  WILCOXON PAREADO (dos colas, 35 réplicas) — previas + nuevas de esta fase")
    print("=" * 78)
    wrows = []
    wrows += _wilcoxon_report("AUC-PR", wide, WILCOXON_AUC_PAIRS)
    wrows += _wilcoxon_report("fpr_latent", wide_fl, WILCOXON_FL_PAIRS)
    wdf = pd.DataFrame(wrows)

    # chequeo (aviso, NO abortar): las comparaciones previas se recalculan sobre
    # los MISMOS datos por réplica; en AUC-PR reproducen exacto, en fpr_latent el
    # test de Wilcoxon sobre valores discretizados (k/6) es sensible a empates a
    # nivel de float y puede moverse ~1e-2 en p (sin cambiar ninguna conclusión).
    if prev_wilcox is not None:
        key = ["metric", "A", "B"]
        merged = prev_wilcox.merge(wdf, on=key, suffixes=("_old", "_new"))
        if len(merged):
            dd = (merged["p_old"] - merged["p_new"]).abs()
            dmax = float(dd.max())
            print(f"\n  [chequeo] {len(merged)} comparaciones previas recalculadas: "
                  f"max|Δp| = {dmax:.2e}")
            if dmax >= 1e-6:
                worst = merged.loc[dd.idxmax()]
                print(f"           (mayor desvío: {worst['metric']} {worst['A']} vs {worst['B']}: "
                      f"p {worst['p_old']:.4g} -> {worst['p_new']:.4g}; "
                      f"sensibilidad a empates float en fpr_latent, conclusión sin cambios)")
            if dmax >= 5e-2:
                print("           AVISO: desvío > 0.05, revisar manualmente.")

    wdf.to_csv(wilcox_csv, index=False)

    print("\n" + "-" * 78)
    if notes:
        print(f"  ADVERTENCIAS / FALLOS {NEW}:")
        for ln in notes:
            print(f"    - {ln}")
    elif not already_done:
        print(f"  {NEW}: sin advertencias de no-convergencia ni fallos registrados.")
    print("=" * 78)
    print(f"\n  Actualizado -> {per_rep_csv}")
    print(f"  Actualizado -> {summ_csv}")
    print(f"  Actualizado -> {wilcox_csv}")
    print(f"  Backups -> phase2_*_5methods_backup.csv, phase2_wilcoxon_prev_backup.csv")


# =====================================================================
# phi_bagged (bagged_phi_network) is included here so that Table 1's
# stability comparison and the ground-truth validation compare the exact
# same set of methods. Uses the SAME n_bootstrap/sample_fraction as the
# other two bagged methods in THIS script (bootstrap_mean/bootstrap_stat use
# n_bootstrap=30; l1_logistic_bagged uses n_bootstrap=30, sample_fraction=0.6
# -- deliberately not the pipeline's own production 75/50, for internal
# coherence of this validation script specifically).
# =====================================================================
PHI_BAGGED_NBOOT = 30
PHI_BAGGED_FRAC = 0.6


def _phi_bagged_score(bin_df):
    """Matriz de score (n x n) de phi_bagged (bagged_phi_network)."""
    from inference_functions import bagged_phi_network
    warnings.filterwarnings("ignore")
    n = bin_df.shape[0]
    try:
        consensus_df, _ = bagged_phi_network(
            bin_df, n_bootstrap=PHI_BAGGED_NBOOT, sample_fraction=PHI_BAGGED_FRAC,
            seed=0, n_jobs=-2,
        )
        return np.asarray(consensus_df.values, dtype=float), None
    except Exception as e:  # pragma: no cover
        return np.full((n, n), np.nan), f"ERROR: {e}"


# ---------------------------------------------------------------------
# =====================================================================
# FULL RUN — all seven methods in one pass  (portable entry point)
# =====================================================================
ALL_METHODS = ["vae", "bootstrap_mean", "bootstrap_stat", "phi_raw",
               "l1_signed", "l1_logistic_bagged", "phi_bagged"]

# Primary criterion of the manuscript: fpr_latent, VAE vs each competitor.
# phi_bagged is a peer bagged competitor reported in Table 1 (alongside
# l1_logistic_bagged), so it gets the same primary comparison as every
# other method in that table.
PRIMARY_FPR_LATENT_PAIRS = [
    ("vae", "phi_raw"),
    ("vae", "l1_signed"),
    ("vae", "l1_logistic_bagged"),
    ("vae", "bootstrap_mean"),
    ("vae", "phi_bagged"),
]


def _all_method_scores(bin_df):
    """dict method -> (n x n) signed score matrix, for all seven methods."""
    scores, notes = _method_scores(bin_df)          # vae, bootstrap_mean, phi_raw, l1_signed
    S_bs, err_bs = _bootstrap_stat_score(bin_df)
    if err_bs:
        notes["bootstrap_stat"] = err_bs
    scores["bootstrap_stat"] = S_bs
    S_l1b, err_l1b = _l1_bagged_score(bin_df)
    if err_l1b:
        notes["l1_logistic_bagged"] = err_l1b
    scores["l1_logistic_bagged"] = S_l1b
    S_phib, err_phib = _phi_bagged_score(bin_df)
    if err_phib:
        notes["phi_bagged"] = err_phib
    scores["phi_bagged"] = S_phib
    return scores, notes


def run_full():
    from statsmodels.stats.multitest import multipletests

    real = load_real_matrix()
    print("=" * 78)
    print("  GROUND-TRUTH VALIDATION — FULL RUN (6 methods)")
    print("=" * 78)
    print(f"  Seeds            : {PHASE2_SEEDS[0]}..{PHASE2_SEEDS[-1]} "
          f"({len(PHASE2_SEEDS)} replicates)")
    print(f"  Simulation kwargs: {SIM_KWARGS}")
    print(f"  Methods          : {', '.join(ALL_METHODS)}")
    print("-" * 78)

    records, all_notes = [], []
    for rep, seed in enumerate(PHASE2_SEEDS, 1):
        bin_df, gt = generate_synthetic_ground_truth(
            seed=seed, real_matrix=real, **SIM_KWARGS)
        n = bin_df.shape[0]
        scores, notes = _all_method_scores(bin_df)
        for m, msg in notes.items():
            all_notes.append(f"seed {seed} / {m}: {msg}")
        for m in ALL_METHODS:
            auc, fpr_i, fpr_l, failed = _eval_score_matrix(scores[m], gt, n)
            records.append(dict(seed=seed, method=m, auc_pr=auc,
                                fpr_indirect=fpr_i, fpr_latent=fpr_l, failed=failed))
        print(f"  replicate {rep:>2}/{len(PHASE2_SEEDS)} (seed {seed}) done")

    res = pd.DataFrame.from_records(records)
    # Round to 12 decimals before writing: this is far below any meaningful
    # precision for these metrics, but it takes every value out of the
    # 16th/17th-significant-digit zone where pandas.read_csv()'s default
    # (non round_trip) float parser can silently return a different float64 --
    # so the CSV round-trips exactly with plain
    # pd.read_csv() with no special arguments (confirmed: 505/505 exact vs.
    # 419/505 unrounded, in a dedicated test).
    for _col in ("auc_pr", "fpr_indirect", "fpr_latent"):
        res[_col] = res[_col].round(12)
    res.to_csv(os.path.join(OUT_DIR, "05_ground_truth_per_replicate_results.csv"), index=False)

    # ---- re-read the file we just wrote, and use THAT as the one and only
    # source for everything downstream (summary + Wilcoxon).
    #
    # pandas.read_csv()'s default C-engine float parser is NOT round-trip-exact:
    # in a spot check it altered ~17% of float64 values by 1 ULP (e.g.
    # "0.027586206896551724" is read back as 0.0275862068965517, a different
    # float64). That is too small to change any sign/threshold decision
    # elsewhere in this pipeline, but scipy.stats.wilcoxon's tie handling is
    # very sensitive to it: two replicates whose true difference is exactly tied
    # can stop being tied after a lossy round-trip, changing the rank sum (W)
    # and therefore p. Results computed from the in-memory data would then not
    # be reproducible by reading the CSV back with plain pd.read_csv().
    #
    # Reading with float_precision="round_trip" (pandas' exact parser; it
    # recovered 505/505 known float64 values exactly vs. 419/505 for the default
    # parser) and re-deriving `res` from the file makes the file on disk and
    # wilcoxon_tests.csv provably the same computation.
    res = pd.read_csv(os.path.join(OUT_DIR, "05_ground_truth_per_replicate_results.csv"),
                       float_precision="round_trip")

    summary = (res[~res["failed"]]
               .groupby("method")[["auc_pr", "fpr_indirect", "fpr_latent"]]
               .agg(["mean", "std", "count"]))
    # same CSV-round-trip safety as wilcoxon_tests.csv below: mean/std are
    # arbitrary continuous floats, so round to 10 significant figures.
    for _c in summary.columns:
        if _c[1] in ("mean", "std"):
            summary[_c] = summary[_c].apply(lambda x: float(f"{x:.10g}") if pd.notna(x) else x)
    summary.to_csv(os.path.join(OUT_DIR, "05_ground_truth_summary.csv"))

    print("\n" + "=" * 78)
    print("  SUMMARY (mean +/- sd over replicates)")
    print("=" * 78)
    for m in ALL_METHODS:
        sub = res[(res["method"] == m) & (~res["failed"])]
        print(f"  {m:<20} AUC-PR={sub['auc_pr'].mean():.3f}+/-{sub['auc_pr'].std():.3f}"
              f"  FPR_ind={sub['fpr_indirect'].mean():.3f}+/-{sub['fpr_indirect'].std():.3f}"
              f"  FPR_lat={sub['fpr_latent'].mean():.3f}+/-{sub['fpr_latent'].std():.3f}"
              f"  (n={len(sub)})")

    # ---- primary criterion: Wilcoxon + Holm-Bonferroni on fpr_latent ----
    wide_fl = res.pivot_table(index="seed", columns="method", values="fpr_latent")
    rows = []
    for A, B in PRIMARY_FPR_LATENT_PAIRS:
        r = _paired_wilcoxon(wide_fl[A].values, wide_fl[B].values)
        rows.append(dict(metric="fpr_latent", A=A, B=B, n=r["n"],
                         median_a=r["median_a"], median_b=r["median_b"],
                         median_diff=r["median_diff"],
                         n_a_gt_b=r["n_a_gt_b"], n_b_gt_a=r["n_b_gt_a"],
                         n_ties=r["n_ties"], W_stat=r["stat"], p_raw=r["p"]))
    wdf = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(wdf["p_raw"].values, alpha=0.05, method="holm")
    wdf["p_holm"] = p_holm
    wdf["significant_holm"] = np.where(reject, "significant", "not_significant")

    # Same CSV-round-trip precision issue as above, but p-values (unlike
    # fpr_latent, which is always an exact multiple of 1/6) are arbitrary
    # continuous floats, so they can't be rounded to a small exact set --
    # round to 10 significant figures instead (confirmed safe for pandas'
    # default reader: 350/350 exact in a dedicated test across realistic
    # p-value magnitudes). No manuscript ever cites a p-value to more than
    # 10 significant figures, so nothing meaningful is lost.
    for _col in ("median_a", "median_b", "median_diff", "W_stat", "p_raw", "p_holm"):
        wdf[_col] = wdf[_col].apply(lambda x: float(f"{x:.10g}") if pd.notna(x) else x)

    # ---- self-check: a SECOND, independent round_trip re-read of the same
    # file must reproduce this exactly. This is now expected to always pass
    # (round_trip parsing is exact and deterministic); it guards against any
    # future regression -- e.g. someone dropping float_precision="round_trip"
    # from a read_csv call -- being caught immediately instead of silently
    # shipping an unreproducible wilcoxon_tests.csv. ----
    _res_check = pd.read_csv(os.path.join(OUT_DIR, "05_ground_truth_per_replicate_results.csv"),
                             float_precision="round_trip")
    _wide_check = _res_check.pivot_table(index="seed", columns="method", values="fpr_latent")
    _mismatches = []
    def _round_sig(x, n=10):
        return float(f"{x:.{n}g}") if pd.notna(x) else x

    for _i, (_A, _B) in enumerate(PRIMARY_FPR_LATENT_PAIRS):
        _r_check = _paired_wilcoxon(_wide_check[_A].values, _wide_check[_B].values)
        _check_p, _check_w = _round_sig(_r_check["p"]), _round_sig(_r_check["stat"])
        _live = wdf.iloc[_i]
        _same_p = (_check_p == _live["p_raw"]) or (np.isnan(_check_p) and np.isnan(_live["p_raw"]))
        _same_w = (_check_w == _live["W_stat"]) or (np.isnan(_check_w) and np.isnan(_live["W_stat"]))
        if not (_same_p and _same_w):
            _mismatches.append(
                f"{_A} vs {_B}: computed -> W={_live['W_stat']!r} p={_live['p_raw']!r}  "
                f"|  independent re-read -> W={_r_check['stat']!r} p={_r_check['p']!r}"
            )
    if _mismatches:
        raise RuntimeError(
            "SELF-CHECK FAILED: an independent re-read of "
            "05_ground_truth_per_replicate_results.csv "
            "does not reproduce the Wilcoxon test just computed. Refusing to write "
            "wilcoxon_tests.csv (it would not be reproducible by a reviewer).\n  "
            + "\n  ".join(_mismatches)
        )
    print("  [self-check] wilcoxon_tests.csv reproducible from "
          "05_ground_truth_per_replicate_results.csv: OK")

    wdf.to_csv(os.path.join(OUT_DIR, "TableS5.csv"), index=False)

    print("\n" + "=" * 78)
    print("  PRIMARY CRITERION — Wilcoxon signed-rank on fpr_latent, Holm-Bonferroni")
    print("=" * 78)
    print(f"  {'A vs B':<34} {'p_raw':>12} {'p_holm':>12}   verdict")
    for _, w in wdf.iterrows():
        print(f"  {w['A'] + ' vs ' + w['B']:<34} {w['p_raw']:>12.4g} {w['p_holm']:>12.4g}"
              f"   {w['significant_holm']}")

    if all_notes:
        print("\n  WARNINGS / FAILURES:")
        for ln in all_notes:
            print(f"    - {ln}")

    print(f"\n  -> {os.path.join(OUT_DIR, '05_ground_truth_per_replicate_results.csv')}")
    print(f"  -> {os.path.join(OUT_DIR, '05_ground_truth_summary.csv')}")
    print(f"  -> {os.path.join(OUT_DIR, 'TableS5.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=str, default="full",
                    choices=("1", "2", "2b", "2c", "full", "select-hyperparams"))
    args = ap.parse_args()

    if args.phase == "select-hyperparams":
        print(">>> POINT 1: VAE hyperparameter search on simulated ground truth "
              "(tuning seeds 200-229). <<<\n")
        run_hyperparameter_search_ground_truth()
    elif args.phase == "full":
        run_full()
    elif args.phase == "1":
        run_pilot()
    elif args.phase == "2":
        print(">>> FASE 2: asumo que el piloto (Fase 1) ya fue revisado y aprobado. <<<\n")
        run_phase2()
    elif args.phase == "2b":
        print(">>> FASE 2b: anadir bootstrap_stat a la Fase 2 ya completada. <<<\n")
        run_phase2b_bootstrap_stat()
    else:
        print(">>> FASE 2c: anadir l1_logistic_bagged a la Fase 2 ya completada. <<<\n")
        run_phase2c_l1_bagged()
