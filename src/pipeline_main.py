"""
pipeline_main.py — main analysis pipeline for the E. coli effector
co-occurrence network.  Run top to bottom (see run_all.py for the full chain).

All paths are relative to the repository root; nothing outside the repo is read
or written.  Outputs go to results/.

Blocks 1-9 build the intermediate artifacts and the Bootstrap-Mean network
(which feeds the Table 1 stability comparison).  The FINAL reported network is
built by build_final_network.py, called at the end of this script.
"""
import os
import sys
import time as _time
from itertools import combinations, product

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, mode

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch  # noqa: E402
torch.set_num_threads(1)  # deterministic + faster on this small VAE

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

_T0 = _time.time()


def _progreso(msg):
    linea = f"[PROGRESO {_time.strftime('%Y-%m-%d %H:%M:%S')} | +{(_time.time() - _T0) / 3600:.2f} h] {msg}"
    print("\n" + linea, flush=True)

from inference_functions import (
    plot_effector_presence_matrix,
    heatmap_coocurrence,
    export_grafo_json_topk,
    cooccurrence_jaccard_rows,
    cooccurrence_phi_fisher_rows,
    bootstrap_vae_network,
    bootstrap_vae_statistical_network,
    plot_bootstrap_std_histogram,
    plot_mean_vs_std_scatter,
    bernoulli_vae_network,
    evaluate_network_empirical,
    neighborhood_selection_logistic,
    bagged_phi_network,
    bagged_l1_network,
    compute_empirical_cooccurrence,
    compute_convergence_metrics,
    plot_bootstrap_convergence,
    normalize_similarity_matrix,
)
from vae_config_loader import load_vae_params


# Block 1: Data loading and binary matrix construction
# -------------------------------------------------------------------
file_path = os.path.join(_ROOT, "data", "input", "EffectorRepertoire_Ecoli.xlsx")
data_output = os.path.join(_ROOT, "results")
figures_output = os.path.join(_ROOT, "results", "figures")
os.makedirs(data_output, exist_ok=True)
os.makedirs(figures_output, exist_ok=True)
sheet_name = "Ecoli"
NORMALIZE_MATRICES = False

print("\n" + "=" * 60)
print("  PIPELINE CONFIGURATION")
print("=" * 60)
print(f"  Matrix normalization : {'[0,1] via p99' if NORMALIZE_MATRICES else 'raw cosine [-1,1]'}")

df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
effector_col = df_raw.columns[2]

# Exclusión robusta de columnas de totales POR NOMBRE, no por posición.
# Antes: strain_cols = df_raw.columns[9:-1]  (recortaba a ciegas la última columna,
# frágil si el orden/nº de columnas de resumen cambia).
candidate_cols = df_raw.columns[9:]
excluded_mask = candidate_cols.str.contains("suma|total", case=False, na=False)
strain_cols = candidate_cols[~excluded_mask]

excluded_cols = list(candidate_cols[excluded_mask])
print(f"Columnas excluidas por el filtro 'suma|total' ({len(excluded_cols)}): {excluded_cols}")

print(f"Columna de efectores: {effector_col}")
print(f"Numero de cepas detectadas: {len(strain_cols)}")

df_raw[effector_col] = df_raw[effector_col].astype(str).str.strip()
df_raw = df_raw[df_raw[effector_col].notna() & (df_raw[effector_col] != "nan") & (df_raw[effector_col] != "")]

counts = {}
unique_names = []
for name in df_raw[effector_col]:
    counts[name] = counts.get(name, 0) + 1
    if counts[name] > 1:
        suffix = f"({chr(96 + counts[name])})"
        new_name = f"{name}{suffix}"
    else:
        new_name = f"{name}(a)" if df_raw[effector_col].tolist().count(name) > 1 else name
    unique_names.append(new_name)

df_raw[effector_col] = unique_names
df = df_raw.copy()

presence_data = df.loc[:, strain_cols].copy()
presence_data = presence_data.applymap(
    lambda x: np.nan if isinstance(x, str) and "pseudogen" in x.lower() else x
)
presence_data.index = df[effector_col]
matrix = presence_data.notna().astype(int)

assert matrix.shape[1] == 150, f"Se esperaban 150 cepas, se obtuvieron {matrix.shape[1]}"
assert matrix.sum(axis=0).min() == 4, "El mínimo de repertorio ya no es 4, revisar antes de continuar"

matrix.to_csv(os.path.join(data_output, "01_binary_strains_effectors.csv"), index=True)


# Block 2: Effector prevalence summary and visualisation
# -------------------------------------------------------------------
cross_species_effectors = ["IpaH?", "DUF1778", "SopA Shigella/NleL-2", "NleL-like?", "OspG/NleH", "OspG"]

effector_count = matrix.sum(axis=1)
effector_count.to_csv(os.path.join(data_output, "01_strains_per_effector.csv"), header=["NumCepas"])
plot_effector_presence_matrix(matrix, cross_species_effectors, figures_output, dpi=600)

total_strains = matrix.shape[1]
threshold_50 = total_strains / 2
total_effectors = len(effector_count)
above_50 = int((effector_count >= threshold_50).sum())
below_50 = int((effector_count < threshold_50).sum())

print("  EFFECTOR PREVALENCE SUMMARY")
print("-" * 60)
print(f"  Total effectors detected : {total_effectors}")
print(f"  Above 50% prevalence     : {above_50}")
print(f"  Below 50% prevalence     : {below_50}")

strains_count = matrix.sum(axis=0)
min_strain = strains_count.idxmin()
n_eff_min = int(strains_count.min())
effectors_present = matrix.index[matrix[min_strain] == 1].tolist()

print("\n")
print("  STRAIN WITH FEWEST EFFECTORS")
print("-" * 60)
print(f"  Strain      : {min_strain}")
print(f"  N effectors : {n_eff_min}")
print(f"  Effectors   : {effectors_present}")

matrix.to_csv(os.path.join(data_output, "01_binary_matrix.csv"))
_progreso(f"BLOQUES 1-2 COMPLETADOS: matriz binaria {matrix.shape[0]}x{matrix.shape[1]}, "
         f"min repertorio={int(matrix.sum(axis=0).min())} ({matrix.sum(axis=0).idxmin()}). "
         f"-> 01_binary_matrix.csv, 01_binary_strains_effectors.csv, 01_strains_per_effector.csv")


# Block 3: VAE hyperparameter search -- SEED-TO-SEED STABILITY (DIAGNOSTIC ONLY)
# -------------------------------------------------------------------
# This block runs the full 48-config x 10-seed grid and writes
# 01_vae_hyperparameter_search_results.csv as a diagnostic / supplementary-table
# artifact (seed-to-seed stability of the fit on real data). It does NOT
# determine the operational `vae_params` used downstream -- stability alone
# does not guarantee an accurate recovery of the true structure. The
# operational hyperparameters for every VAE fit in this script from here on
# (permutation control, outer stability comparison, bootstraps, the
# Bootstrap-Mean network) are loaded from the single shared
# src/vae_config.json (see vae_config_loader.load_vae_params()), which was
# selected in simulation_ground_truth.py by ACCURACY against simulated ground
# truth (fpr_latent, primary criterion), not by stability.
search_space = {
    "latent_dim": [8, 16, 32],
    "hidden_dim": [8, 16, 32, 64],
    "beta_kl": [0.01, 0.05, 0.1, 0.5],
}


def generate_param_grid(space):
    keys = list(space.keys())
    values = list(space.values())
    return [dict(zip(keys, vals)) for vals in product(*values)]


param_grid = generate_param_grid(search_space)
vae_base_params = {
    "epochs": 300,
    "batch_size": 32,
    "lr": 1e-3,
    "gamma_max": 1e-3,
    "burn_in_frac": 0.3,
    "ramp_frac": 0.3,
}


def upper_triangle_vector(adj_df):
    values = np.asarray(adj_df, dtype=float)
    return values[np.triu_indices_from(values, k=1)]


print("\n" + "=" * 60)
print("  VAE HYPERPARAMETER SEARCH")
print("=" * 60)
print(f"  Combinations to evaluate : {len(param_grid)}")
print("-" * 60)

results_rows = []
for config_idx, params in enumerate(param_grid, start=1):
    print(
        f"  Config {config_idx:02d}/{len(param_grid)} | "
        f"latent_dim={params['latent_dim']}, "
        f"hidden_dim={params['hidden_dim']}, "
        f"beta_kl={params['beta_kl']}"
    )

    vectors = []
    for seed in range(10):
        adj_df, _, _ = bernoulli_vae_network(
            matrix,
            normalize_adj=NORMALIZE_MATRICES,
            verbose=False,
            seed=seed,
            **vae_base_params,
            **params,
        )
        vectors.append(upper_triangle_vector(adj_df))

    pairwise_r = [
        pearsonr(vectors[i], vectors[j])[0]
        for i, j in combinations(range(len(vectors)), 2)
    ]
    pairwise_r = np.asarray(pairwise_r, dtype=float)

    mean_stability_r = float(np.mean(pairwise_r))
    std_stability_r = float(np.std(pairwise_r))
    results_rows.append({
        **params,
        "mean_stability_r": mean_stability_r,
        "std_stability_r": std_stability_r,
    })
    print(f"    mean_stability_r={mean_stability_r:.6f}, std_stability_r={std_stability_r:.6f}")

hyperparameter_results = pd.DataFrame(
    results_rows,
    columns=[
        "latent_dim",
        "hidden_dim",
        "beta_kl",
        "mean_stability_r",
        "std_stability_r",
    ],
).sort_values("mean_stability_r", ascending=False)
hyperparameter_results.to_csv(
    os.path.join(data_output, "01_vae_hyperparameter_search_results.csv"),
    index=False,
)

best_row = hyperparameter_results.iloc[0]
best = {
    "params": {
        "latent_dim": int(best_row["latent_dim"]),
        "hidden_dim": int(best_row["hidden_dim"]),
        "beta_kl": float(best_row["beta_kl"]),
    },
    "mean_stability_r": float(best_row["mean_stability_r"]),
    "std_stability_r": float(best_row["std_stability_r"]),
}

print("\nBest VAE hyperparameters:")
print(best["params"])
print(f"  mean_stability_r : {best['mean_stability_r']:.6f}")
print(f"  std_stability_r  : {best['std_stability_r']:.6f}")
print("\nSaved hyperparameter search results to "
      + os.path.join(data_output, "01_vae_hyperparameter_search_results.csv"))
_progreso(f"BLOQUE 3 COMPLETADO: VAE hyperparameter search (48 configs x 10 seeds). "
         f"Best={best['params']}, mean_stability_r={best['mean_stability_r']:.4f}. "
         f"-> 01_vae_hyperparameter_search_results.csv")

if "matrix" not in globals():
    matrix = pd.read_csv(
        os.path.join(data_output, "01_binary_matrix.csv"),
        index_col=0,
    )

matrix_values = matrix.to_numpy(dtype=float)
emp_values = matrix_values @ matrix_values.T / matrix_values.shape[1]
upper_idx = np.triu_indices_from(emp_values, k=1)
emp_upper = emp_values[upper_idx]
null_prediction = np.mean(emp_upper)
null_mae = np.mean(np.abs(emp_upper - null_prediction))
print(f"Null model MAE: {null_mae:.6f}")

# vae_params from here on is the single shared ground-truth-selected
# configuration (src/vae_config.json), NOT `best` (the stability-based
# winner computed above, kept only as a diagnostic).
vae_params = load_vae_params()
print("\nOperational VAE hyperparameters (from src/vae_config.json, "
      "ground-truth-selected):")
print(vae_params)
if vae_params.get("beta_kl") != best["params"]["beta_kl"] or \
   vae_params.get("latent_dim") != best["params"]["latent_dim"] or \
   vae_params.get("hidden_dim") != best["params"]["hidden_dim"]:
    print("  NOTE: differs from the stability-based 'best' above "
          f"({best['params']}) -- expected, this block is diagnostic-only "
          "(see the Block 3 comment above).")

# Block 4: Permutation control
# -------------------------------------------------------------------
# Both arms of this permutation control must use the OPERATIONAL vae_params
# from src/vae_config.json -- the real arm's stability numbers are computed
# fresh here rather than reused from Block 3's diagnostic stability search
# (which is fit with a different, stability-selected configuration), so both
# arms are fit with the same VAE configuration.
#
# Design: both arms use the SAME 10 VAE seeds (0-9), so VAE-initialization
# variability is identical and paired 1-to-1 across arms (real replicate i
# and permuted replicate i always use the same VAE seed=i). The only
# remaining difference between arms is the data itself (unpermuted for the
# real arm; row-permuted with permutation-seed=200+i for permuted replicate
# i). This residual data-permutation variability in the null arm is not a
# design flaw -- it is what a permutation null necessarily requires (a
# single arbitrary permuted matrix would not represent the null
# distribution; independent draws from it are needed). The
# model-initialization source of variability is identical and paired across
# both arms, so the only thing that differs between arms is the property
# under test: whether the data has real co-occurrence structure or not.
print("\n" + "=" * 60)
print("  PERMUTATION CONTROL")
print("=" * 60)
print("  Symmetric design: both arms use the SAME 10 VAE seeds (0-9), paired")
print("  1-to-1 (real replicate i and permuted replicate i share seed=i).")
print("  Real arm    : fixed real matrix,      VAE seed varies (0-9).")
print("  Permuted arm: row-permuted matrix (independent draw per replicate,")
print("                permutation-seed=200+i), VAE seed varies (0-9, SAME")
print("                seeds as the real arm) -- the residual data variation")
print("                is inherent to what a permutation null requires, not")
print("                an uncontrolled asymmetry.")

real_vectors = []
for seed in range(10):
    adj_real, _, _ = bernoulli_vae_network(
        matrix,
        normalize_adj=NORMALIZE_MATRICES,
        verbose=False,
        seed=seed,
        **vae_params,
    )
    real_vectors.append(upper_triangle_vector(adj_real))

real_pairwise_r = np.asarray([
    pearsonr(real_vectors[i], real_vectors[j])[0]
    for i, j in combinations(range(len(real_vectors)), 2)
], dtype=float)

perm_vectors = []
for seed in range(10):
    random_state = 200 + seed
    rng_perm = np.random.default_rng(random_state)
    mat_perm = matrix.copy()
    # Permutación POR FILA (por efector): baraja qué cepas portan cada efector,
    # conservando la prevalencia marginal de cada efector (suma por fila) pero
    # destruyendo la co-ocurrencia entre efectores. Antes se permutaba por
    # columna, lo que conservaba el tamaño de repertorio por cepa pero NO la
    # prevalencia por efector, en contra de lo que afirma el manuscrito.
    for idx in mat_perm.index:
        mat_perm.loc[idx] = rng_perm.permutation(mat_perm.loc[idx].values)

    assert mat_perm.sum(axis=1).equals(matrix.sum(axis=1)), (
        "La permutación por fila no conserva la prevalencia marginal por efector"
    )

    adj_perm, _, _ = bernoulli_vae_network(
        mat_perm,
        normalize_adj=NORMALIZE_MATRICES,
        verbose=False,
        seed=seed,  # SAME seed value as the paired real-arm replicate above
        **vae_params,
    )
    perm_vectors.append(upper_triangle_vector(adj_perm))

perm_pairwise_r = np.asarray([
    pearsonr(perm_vectors[i], perm_vectors[j])[0]
    for i, j in combinations(range(len(perm_vectors)), 2)
], dtype=float)

mean_r_real = float(np.mean(real_pairwise_r))
std_r_real = float(np.std(real_pairwise_r))
mean_r_perm = float(np.mean(perm_pairwise_r))
std_r_perm = float(np.std(perm_pairwise_r))

print(f"\n  Stability (real data, seed variation):       {mean_r_real:.6f} +/- {std_r_real:.6f}")
print(f"  Stability (permuted data, structure absent): {mean_r_perm:.6f} +/- {std_r_perm:.6f}")

pd.DataFrame([{
    "mean_r_real": mean_r_real,
    "std_r_real": std_r_real,
    "mean_r_perm": mean_r_perm,
    "std_r_perm": std_r_perm,
    "vae_params": str(vae_params),
    "design": "symmetric: same 10 VAE seeds (0-9) paired 1-to-1 across both "
             "arms; real arm = fixed real matrix; permuted arm = independent "
             "row-permutation per replicate (seed 200+i)",
}]).to_csv(
    os.path.join(data_output, "01_permutation_control_results_rowwise.csv"),
    index=False,
)
_progreso(f"BLOQUE 4 COMPLETADO: control de permutacion POR FILA, diseno simetrico "
         f"(mismas 10 semillas VAE en ambos brazos). "
         f"stability real={mean_r_real:.4f}, permutado={mean_r_perm:.4f}. "
         f"-> 01_permutation_control_results_rowwise.csv")


# ── Block 5a: Inner bootstrap convergence diagnostic ──
# Reassigned explicitly (same shared-config object already loaded above) so
# this block reads correctly standalone.
vae_params = load_vae_params()

B_list = [10, 25, 50, 75, 100]
fractions = [0.5, 0.6, 0.8]

convergence_results = []

for B in B_list:
    for frac in fractions:
        _, _, S_boot, _ = bootstrap_vae_network(
            matrix=matrix,
            n_bootstrap=B,
            sample_fraction=frac,
            vae_params=vae_params,
            normalize_for_stats=NORMALIZE_MATRICES,
            verbose=False,
        )
        S_boot_norm = normalize_similarity_matrix(S_boot, normalize=NORMALIZE_MATRICES)
        mean_corr_curve, std_corr_curve, rmse_mean_curve = compute_convergence_metrics(S_boot_norm)
        convergence_results.append({
            "B": B,
            "sample_fraction": frac,
            "mean_corr_curve": mean_corr_curve,
            "std_corr_curve": std_corr_curve,
            "rmse_mean_curve": rmse_mean_curve,
        })


# Guardar convergence_results para reutilizar sin reejecutar el loop
import json

_conv_rows = []
for r in convergence_results:
    _conv_rows.append({
        "B": r["B"],
        "sample_fraction": r["sample_fraction"],
        "mean_corr_curve":  json.dumps([float(x) for x in r["mean_corr_curve"]]),
        "std_corr_curve":   json.dumps([float(x) for x in r["std_corr_curve"]]),
        "rmse_mean_curve":  json.dumps([float(x) for x in r["rmse_mean_curve"]]),
    })

pd.DataFrame(_conv_rows).to_csv(
    os.path.join(data_output, "01_bootstrap_convergence_results.csv"),
    index=False
)

plot_bootstrap_convergence(
    results=convergence_results,
    selected_B=75,
    selected_frac=0.6,
    figures_output=figures_output,
    dpi=600,
)
_progreso("BLOQUE 5a COMPLETADO: convergencia bootstrap interno (B x fraccion). "
         "-> 01_bootstrap_convergence_results.csv + figuras de convergencia")

# Block 5: Outer stability comparison and single-run VAE edge analysis
# -------------------------------------------------------------------
# Reassigned explicitly (same shared-config object already loaded above) so
# this block reads correctly standalone.
vae_params = load_vae_params()


def upper_triangle_values(arr):
    ii, jj = np.triu_indices(arr.shape[0], k=1)
    return arr[ii, jj]


def top_edge_set(arr, k=50):
    ii, jj = np.triu_indices(arr.shape[0], k=1)
    values = np.abs(arr[ii, jj])
    top_idx = np.argsort(values)[-k:]
    return {tuple(sorted((int(ii[idx]), int(jj[idx])))) for idx in top_idx}


def pairwise_stability(matrices, top_k=50):
    corr_values = []
    jaccard_values = []
    for a_idx, b_idx in combinations(range(len(matrices)), 2):
        a = matrices[a_idx]
        b = matrices[b_idx]
        a_flat = upper_triangle_values(a)
        b_flat = upper_triangle_values(b)
        corr = pearsonr(a_flat, b_flat)[0]
        if np.isfinite(corr):
            corr_values.append(corr)
        a_top = top_edge_set(a, k=top_k)
        b_top = top_edge_set(b, k=top_k)
        union = a_top | b_top
        jaccard_values.append(len(a_top & b_top) / len(union) if union else np.nan)
    return {
        "pearson_mean": float(np.nanmean(corr_values)),
        "pearson_std": float(np.nanstd(corr_values)),
        "top50_jaccard_mean": float(np.nanmean(jaccard_values)),
        "top50_jaccard_std": float(np.nanstd(jaccard_values)),
    }


print("\n" + "=" * 60)
print("  OUTER STABILITY COMPARISON")
print("=" * 60)

n_outer = 30
outer_fraction = 0.8
seed = 42
methods = [
    "vae",
    "bootstrap_mean",
    "bootstrap_stat",
    "l1_logistic (signed)",
    "l1_logistic_bagged",
    "phi",
    "phi_bagged",
]
S_all = {method: [] for method in methods}
rng_outer = np.random.default_rng(seed)
n_subsample = int(matrix.shape[1] * outer_fraction)

for i in range(n_outer):
    subsample_idx = rng_outer.choice(matrix.shape[1], size=n_subsample, replace=False)
    subsample = matrix.iloc[:, subsample_idx]
    print(f"  Outer iteration {i + 1}/{n_outer}")

    adj_vae, _, _ = bernoulli_vae_network(
        subsample,
        normalize_adj=NORMALIZE_MATRICES,
        verbose=False,
        **vae_params,
    )
    S_all["vae"].append(adj_vae.values)

    consensus_matrix, _, _, _ = bootstrap_vae_network(
        subsample,
        n_bootstrap=75,
        sample_fraction=0.6,
        vae_params=vae_params,
        normalize_for_stats=NORMALIZE_MATRICES,
        device="cpu",
        seed=seed + i,
        verbose=False,
    )
    S_all["bootstrap_mean"].append(consensus_matrix.values)

    final_adj_df, _, _ = bootstrap_vae_statistical_network(
        subsample,
        vae_params=vae_params,
        n_bootstrap=75,
        sample_fraction=0.6,
        normalize_for_stats=NORMALIZE_MATRICES,
        device="cpu",
        seed=seed + i,
        verbose=False,
    )
    S_all["bootstrap_stat"].append(final_adj_df.values)

    # symm_rule='average_signed' keeps the sign of the L1 coefficient,
    # matching l1_signed (Table 1) and l1_logistic_bagged (both tables) --
    # the default 'average' would discard the sign
    # (adj_ij = (|coef_ij| + |coef_ji|) / 2), making this row of
    # 01_method_stability_comparison_table1_input.csv incoherent with the
    # other two L1 rows. The method label above, 'l1_logistic (signed)',
    # makes this explicit for anyone reading the CSV.
    adj_l1, _ = neighborhood_selection_logistic(
        subsample,
        normalize_adj=NORMALIZE_MATRICES,
        symm_rule="average_signed",
    )
    S_all["l1_logistic (signed)"].append(adj_l1.values)

    l1_bagged_consensus, _ = bagged_l1_network(
        subsample,
        n_bootstrap=50,
        sample_fraction=0.6,
        seed=seed + i,
        n_jobs=-2,
    )
    S_all["l1_logistic_bagged"].append(l1_bagged_consensus.values)

    _, phi_mat, _ = cooccurrence_phi_fisher_rows(
        subsample,
        normalize_adj=False,
    )
    S_all["phi"].append(phi_mat)

    phi_bagged_consensus, _ = bagged_phi_network(
        subsample,
        n_bootstrap=75,
        sample_fraction=0.6,
        seed=seed + i,
        n_jobs=-2,
    )
    S_all["phi_bagged"].append(phi_bagged_consensus.values)

    _progreso(f"  BLOQUE 5 - iteracion outer {i + 1}/{n_outer} completada")

outer_stability_rows = []
for method in methods:
    metrics = pairwise_stability(S_all[method], top_k=50)
    outer_stability_rows.append({"method": method, **metrics})

outer_stability_comparison = pd.DataFrame(outer_stability_rows)
outer_stability_comparison.to_csv(
    os.path.join(data_output, "01_method_stability_comparison_table1_input.csv"),
    index=False,
)
print(outer_stability_comparison)
_progreso("BLOQUE 5 COMPLETADO: outer stability comparison, "
         f"{len(outer_stability_rows)} filas ({', '.join(m['method'] for m in outer_stability_rows)}). "
         "-> 01_method_stability_comparison_table1_input.csv\n"
         + outer_stability_comparison.to_string(index=False))

print("\n" + "=" * 60)
print("  INDIVIDUAL EDGE STABILITY COMPARISON FOR SINGLE RUN VAE")
print("=" * 60)

S_vae = np.array(S_all["vae"])
plot_bootstrap_std_histogram(
    S_vae,
    figures_output,
    'singleVAE',
    dpi=400,
    log_scale=True,
)
plot_mean_vs_std_scatter(
    S_vae,
    figures_output,
    'singleVAE',
    dpi=400,
    k=50,
)

S_vae = np.array(S_all["vae"])
sd_per_edge = S_vae.std(axis=0)[np.triu_indices(63, k=1)]
mu_per_edge = S_vae.mean(axis=0)[np.triu_indices(63, k=1)]
stability_ratio = np.abs(mu_per_edge) / (sd_per_edge + 1e-12)
print(f"Per-edge σ: mode≈{float(mode(np.round(sd_per_edge,2))[0]):.3f}, "
      f"median={np.median(sd_per_edge):.3f}, 95th pct={np.percentile(sd_per_edge,95):.3f}")
print(f"Stability ratio: mean={stability_ratio.mean():.2f}, median={np.median(stability_ratio):.2f}")

# Guardar matrices para reusar sin re-ejecutar
np.save(
    os.path.join(data_output, "01_S_vae_outer.npy"),
    np.array(S_all["vae"])
)
np.save(
    os.path.join(data_output, "01_S_bm_outer.npy"),
    np.array(S_all["bootstrap_mean"])
)

print("\n" + "=" * 60)
print("  INDIVIDUAL EDGE STABILITY COMPARISON FOR BOOTSTRAP MEAN")
print("=" * 60)

S_vae = np.array(S_all["bootstrap_mean"])
plot_bootstrap_std_histogram(
    S_vae,
    figures_output,
    'bootstrap',
    dpi=400,
    log_scale=True,
)
plot_mean_vs_std_scatter(
    S_vae,
    figures_output,
    'bootstrap',
    dpi=400,
    k=50,
)
# Métricas per-edge de Bootstrap Mean
S_bm = np.array(S_all["bootstrap_mean"])
idx = np.triu_indices(S_bm.shape[1], k=1)
sd_bm = S_bm.std(axis=0)[idx]
mu_bm = S_bm.mean(axis=0)[idx]

print(
    f"Bootstrap Mean per-edge σ: "
    f"mode≈{float(mode(np.round(sd_bm, 2))[0]):.3f}, "
    f"median={np.median(sd_bm):.3f}, "
    f"95th pct={np.percentile(sd_bm, 95):.3f}"
)
pd.DataFrame({
    "mu": mu_bm,
    "sigma": sd_bm
}).to_csv(os.path.join(data_output, "01_bootstrap_mean_per_edge_stats.csv"), index=False)
_progreso("BLOQUE 5 (cola) COMPLETADO: per-edge stability VAE y Bootstrap Mean. "
         "-> 01_S_vae_outer.npy, 01_S_bm_outer.npy, 01_bootstrap_mean_per_edge_stats.csv + figuras")


reference_bootstrap_mean, _, _, _ = bootstrap_vae_network(
    matrix,
    n_bootstrap=75,
    sample_fraction=0.6,
    normalize_for_stats=NORMALIZE_MATRICES,
    vae_params=vae_params,
    device="cpu",
    seed=42,
    verbose=False,
)


# Block 6: Bootstrap Mean vs Jaccard differential edge analysis
# -------------------------------------------------------------------
matrix_jaccard_full = cooccurrence_jaccard_rows(
    matrix,
    normalize_adj=NORMALIZE_MATRICES,
)
matrix_jaccard_full = matrix_jaccard_full.reindex(index=matrix.index, columns=matrix.index)

effectors = reference_bootstrap_mean.index.tolist()
_ii, _jj = np.triu_indices(len(effectors), k=1)

# El corte "top 30%" DEBE usar el mismo criterio que export_grafo_json_topk
# (la función que genera la red exportada, effectornetwork_*.json). Ese criterio es:
#   1. rankear por valor en el triángulo superior (|peso| para Bootstrap Mean),
#   2. EXCLUIR los pares con valor exactamente cero antes de cortar,
#   3. seleccionar el top 30% POR CONTEO sobre los no-cero:
#      k = int(len(nonzero) * 30 / 100),
#   4. usar como umbral el k-ésimo mayor valor observado (no np.percentile, que
#      interpola entre estadísticos de orden y sobre una distribución con ceros),
#   5. retener con >= umbral (inclusivo, no > estricto).
# Se aplica por separado a bm_abs_values y a jaccard_values, cada uno con su
# propio conjunto de no-cero y su propio k.
def _top_percent_threshold(values, top_percent=30):
    nonzero = values[values > 0]
    if nonzero.size == 0:
        raise ValueError("No hay valores no nulos para el corte top-k.")
    k = max(int(len(nonzero) * top_percent / 100), 1)
    return float(np.sort(nonzero)[-k])


bm_values = reference_bootstrap_mean.values[_ii, _jj]
bm_abs_values = np.abs(bm_values)
bm_threshold_30 = _top_percent_threshold(bm_abs_values, 30)
edges_bm = {
    frozenset({effectors[_ii[k]], effectors[_jj[k]]})
    for k in np.where((bm_abs_values >= bm_threshold_30) & (bm_abs_values > 0))[0]
}

jaccard_values = matrix_jaccard_full.values[_ii, _jj]
jaccard_threshold_30 = _top_percent_threshold(jaccard_values, 30)
edges_jaccard = {
    frozenset({effectors[_ii[k]], effectors[_jj[k]]})
    for k in np.where((jaccard_values >= jaccard_threshold_30) & (jaccard_values > 0))[0]
}

shared_edges = edges_bm & edges_jaccard
bm_not_jaccard = edges_bm - edges_jaccard
jaccard_not_bm = edges_jaccard - edges_bm

effector_prevalence = effector_count / 150


def mean_edge_prevalence(edge_set):
    if not edge_set:
        return np.nan
    edge_prevalences = []
    for edge in edge_set:
        eff1, eff2 = sorted(edge)
        edge_prevalences.append(
            float((effector_prevalence[eff1] + effector_prevalence[eff2]) / 2)
        )
    return float(np.mean(edge_prevalences))


mean_prev_bm_not_jaccard = mean_edge_prevalence(bm_not_jaccard)
mean_prev_jaccard_not_bm = mean_edge_prevalence(jaccard_not_bm)

print("\n" + "=" * 60)
print("  DIFFERENTIAL EDGE ANALYSIS: BOOTSTRAP MEAN VS JACCARD")
print("=" * 60)
print(f"  Total edges in Bootstrap Mean top 30% : {len(edges_bm)}")
print(f"  Total edges in Jaccard top 30%        : {len(edges_jaccard)}")
print(f"  Edges shared by both methods          : {len(shared_edges)}")
print(f"  Edges in Bootstrap Mean NOT Jaccard   : {len(bm_not_jaccard)}")
print(f"  Edges in Jaccard NOT Bootstrap Mean   : {len(jaccard_not_bm)}")
print(f"  Mean prevalence, BM-only edges        : {mean_prev_bm_not_jaccard:.4f}")
print(f"  Mean prevalence, Jaccard-only edges   : {mean_prev_jaccard_not_bm:.4f}")

bm_not_jaccard_rows = []
for edge in bm_not_jaccard:
    eff1, eff2 = sorted(edge)
    bm_score = float(reference_bootstrap_mean.loc[eff1, eff2])
    jaccard_score = float(matrix_jaccard_full.loc[eff1, eff2])
    mean_prevalence = float((effector_prevalence[eff1] + effector_prevalence[eff2]) / 2)
    bm_not_jaccard_rows.append((eff1, eff2, bm_score, jaccard_score, mean_prevalence))

bm_not_jaccard_rows.sort(key=lambda row: abs(row[2]), reverse=True)

print("\n  Top 20 Bootstrap Mean edges not in Jaccard top 30%")
print(f"  {'Pair':<35} {'BM score':>10} {'Jaccard':>10} {'Mean prev.':>12}")
print(f"  {'-' * 72}")
for eff1, eff2, bm_score, jaccard_score, mean_prevalence in bm_not_jaccard_rows[:20]:
    pair = f"{eff1} - {eff2}"
    print(f"  {pair:<35} {bm_score:>10.4f} {jaccard_score:>10.4f} {mean_prevalence:>12.4f}")

# --- Guardado de resultados (antes este bloque sólo imprimía por stdout) ---
pd.DataFrame([{
    "bm_top30_edges": len(edges_bm),
    "jaccard_top30_edges": len(edges_jaccard),
    "shared_edges": len(shared_edges),
    "bm_only_edges": len(bm_not_jaccard),
    "jaccard_only_edges": len(jaccard_not_bm),
    "mean_prevalence_bm_only": mean_prev_bm_not_jaccard,
    "mean_prevalence_jaccard_only": mean_prev_jaccard_not_bm,
}]).to_csv(os.path.join(data_output, "01_jaccard_differential_summary.csv"), index=False)

_diff_edge_rows = []
for _edge in bm_not_jaccard:
    _e1, _e2 = sorted(_edge)
    _diff_edge_rows.append({
        "eff_a": _e1,
        "eff_b": _e2,
        "set": "bm_only",
        "bm_score": float(reference_bootstrap_mean.loc[_e1, _e2]),
        "jaccard_score": float(matrix_jaccard_full.loc[_e1, _e2]),
    })
for _edge in jaccard_not_bm:
    _e1, _e2 = sorted(_edge)
    _diff_edge_rows.append({
        "eff_a": _e1,
        "eff_b": _e2,
        "set": "jaccard_only",
        "bm_score": float(reference_bootstrap_mean.loc[_e1, _e2]),
        "jaccard_score": float(matrix_jaccard_full.loc[_e1, _e2]),
    })
pd.DataFrame(
    _diff_edge_rows,
    columns=["eff_a", "eff_b", "set", "bm_score", "jaccard_score"],
).sort_values(["set", "bm_score"], ascending=[True, False]).to_csv(
    os.path.join(data_output, "01_jaccard_differential_edges.csv"), index=False
)

_progreso(f"BLOQUE 6 COMPLETADO: differential edges Bootstrap Mean vs Jaccard "
         f"(BM top30%={len(edges_bm)}, Jaccard top30%={len(edges_jaccard)}, "
         f"solo-BM={len(bm_not_jaccard)}, solo-Jaccard={len(jaccard_not_bm)}). "
         f"-> 01_jaccard_differential_summary.csv, 01_jaccard_differential_edges.csv")


# Block 7: Final network construction (Bootstrap Mean, all 150 strains)
# -------------------------------------------------------------------
selected_method = "Bootstrap Mean"

print("\n" + "=" * 60)
print("  FINAL NETWORK CONSTRUCTION")
print("=" * 60)
print(f"  Method : {selected_method}")
print("-" * 60)

final_matrix, _, S_boot_final, _ = bootstrap_vae_network(
    matrix,
    n_bootstrap=75,
    sample_fraction=0.6,
    normalize_for_stats=NORMALIZE_MATRICES,
    vae_params=vae_params,
    device="cpu",
    seed=42,
    verbose=False,
)

heatmap_coocurrence(final_matrix, selected_method, figures_output, dpi=600)

edge_score_matrix = np.abs(final_matrix.values) if not NORMALIZE_MATRICES else final_matrix.values
tri_vals = edge_score_matrix[np.triu_indices(len(final_matrix), k=1)]
nonzero_tri_vals = tri_vals[tri_vals > 0]
k_edges = max(int(len(nonzero_tri_vals) * 30 / 100), 1)
edge_threshold = np.sort(nonzero_tri_vals)[-k_edges]
n_edges = int(np.sum(tri_vals >= edge_threshold))
_progreso(f"BLOQUE 7 COMPLETADO: red final Bootstrap Mean sobre 150 cepas "
         f"({len(final_matrix)} nodos). -> heatmap en results/figures")


# Block 8: Threshold sensitivity analysis
# -------------------------------------------------------------------
# El criterio de corte DEBE ser idéntico al del Bloque 7 / export_grafo_json_topk
# (la red final exportada, effectornetwork_*.json). Ese criterio es:
#   1. rankear por |peso| en el triángulo superior (k=1),
#   2. EXCLUIR los pares con peso exactamente cero antes de cortar,
#   3. seleccionar el top (100 - percentil)% POR CONTEO sobre los valores no-cero,
#   4. usar como umbral el k-ésimo mayor valor observado (no np.percentile, que
#      interpola entre estadísticos de orden y sobre una distribución con ceros),
#   5. retener las aristas con |peso| >= umbral (inclusivo, no > estricto).
# Con este criterio el percentil 70 reproduce EXACTAMENTE la red exportada
# (585 aristas, |peso| mínimo ~0.3196).
threshold_percentiles = [60, 65, 68, 69, 70, 71, 72, 75, 80]
final_abs_values = np.abs(final_matrix.values) if not NORMALIZE_MATRICES else final_matrix.values
upper_idx = np.triu_indices_from(final_abs_values, k=1)
upper_abs_values = final_abs_values[upper_idx]
nonzero_sorted = np.sort(upper_abs_values[upper_abs_values > 0])

threshold_rows = []
for pct in threshold_percentiles:
    k_edges_pct = max(int(len(nonzero_sorted) * (100 - pct) / 100), 1)
    threshold_pct = float(nonzero_sorted[-k_edges_pct])
    adjacency = ((final_abs_values >= threshold_pct) & (final_abs_values > 0)).astype(int)
    np.fill_diagonal(adjacency, 0)
    degree = adjacency.sum(axis=1)
    n_edges_pct = int(adjacency[upper_idx].sum())
    n_isolated = int(np.sum(degree == 0))
    min_degree = int(degree.min())

    threshold_rows.append({
        "percentile": pct,
        "threshold": round(threshold_pct, 6),
        "n_edges": n_edges_pct,
        "n_isolated": n_isolated,
        "min_degree": min_degree,
    })

threshold_sensitivity = pd.DataFrame(threshold_rows)
threshold_sensitivity.to_csv(
    os.path.join(data_output, "01_threshold_sensitivity_bootstrap_mean.csv"),
    index=False,
)
_progreso("BLOQUE 8 COMPLETADO: threshold sensitivity. -> 01_threshold_sensitivity_bootstrap_mean.csv")

print("\n" + "=" * 60)
print("  THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 60)
print(threshold_sensitivity.to_string(index=False))

connected_rows = threshold_sensitivity[threshold_sensitivity["n_isolated"] == 0]
if not connected_rows.empty:
    inflection_row = connected_rows.iloc[-1]
    inflection_pct = int(inflection_row["percentile"])
    inflection_min_degree = int(inflection_row["min_degree"])
    print(
        f"\nInflection point: percentile {inflection_pct} is the maximum threshold at which "
        f"all 63 effector families retain at least one connection "
        f"(min_degree = {inflection_min_degree}, n_isolated = 0)."
    )

    next_rows = threshold_sensitivity[threshold_sensitivity["percentile"] > inflection_pct]
    if not next_rows.empty:
        next_row = next_rows.iloc[0]
        print(
            f"At percentile {int(next_row['percentile'])}: "
            f"{int(next_row['n_isolated'])} effectors become isolated."
        )
else:
    first_row = threshold_sensitivity.iloc[0]
    print(
        "\nInflection point: no tested percentile retains all 63 effector families "
        "with at least one connection."
    )
    print(
        f"At percentile {int(first_row['percentile'])}: "
        f"{int(first_row['n_isolated'])} effectors are isolated."
    )


# Block 9: Network export (JSON)
# -------------------------------------------------------------------
export_grafo_json_topk(
    final_matrix,
    effector_count,
    total_strains,
    data_output,
    top_percent=30,
    filename="bootstrap_mean",
    use_absolute_weights=not NORMALIZE_MATRICES,
)
# canonical repo name for the Bootstrap-Mean network (kept alongside the
# reported single-shot network; feeds the Table 1 stability comparison)
os.replace(
    os.path.join(data_output, "effectornetwork_top30_bootstrap_mean.json"),
    os.path.join(data_output, "01_network_bootstrap_mean_top30_table1_input.json"),
)
_progreso("BLOQUE 9 COMPLETADO: export JSON. -> 01_network_bootstrap_mean_top30_table1_input.json")


# Block 10: Final summary
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Selected method          : {selected_method}")
print(f"  Nodes in network         : {len(final_matrix)}")
print(f"  Edges retained (top 30%) : {n_edges}")
print(f"  Edge threshold (top 30%) : {edge_threshold:.6f}")
_progreso(f"PIPELINE (blocks 1-10) COMPLETADO. Nodos={len(final_matrix)}, "
         f"edges top30%={n_edges}, umbral={edge_threshold:.6f}. "
         f"Tiempo total: {(_time.time() - _T0) / 3600:.2f} h")


# Block 11: Final reported network (single-shot VAE + bootstrap confidence)
# -------------------------------------------------------------------
# The Bootstrap-Mean network above feeds the Table 1 stability comparison.
# The network REPORTED IN THE MANUSCRIPT is the single-shot VAE fit on the full
# 150-strain matrix, with per-edge confidence from 75 bootstrap replicas.
# run_all.py runs build_final_network.py as its own step, so it sets
# SKIP_BUILD_FINAL_NETWORK=1 to avoid doing this ~2 h computation twice.
if os.environ.get("SKIP_BUILD_FINAL_NETWORK") == "1":
    print("\n[pipeline_main] SKIP_BUILD_FINAL_NETWORK=1 -> build_final_network.py "
          "will be run as a separate step.")
else:
    print("\n" + "=" * 60)
    print("  HANDING OFF TO build_final_network.py")
    print("=" * 60)
    import build_final_network  # noqa: E402
    build_final_network.main()
