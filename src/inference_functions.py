import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import json
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import math
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

import seaborn as sns
# sns.set_style("white")

# plt.rcParams.update({
#     "font.family": "Arial",
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

def normalize_similarity_matrix(values, normalize=True, quantile=0.99, clip=True,
                                fill_diagonal=True, norm_factor=None):
    """
    Normaliza matrices de similitud con un criterio común.

    Si normalize=True, escala por un cuantil alto de los valores positivos y
    opcionalmente recorta a [0,1]. Si normalize=False, devuelve los valores
    crudos preservando su escala original.
    """
    arr = np.asarray(values, dtype=float).copy()

    def _off_diagonal_values(x):
        if x.ndim == 2 and x.shape[0] == x.shape[1]:
            mask = ~np.eye(x.shape[0], dtype=bool)
            return x[mask]
        if x.ndim == 3 and x.shape[1] == x.shape[2]:
            mask = ~np.eye(x.shape[1], dtype=bool)
            return x[:, mask]
        return x.reshape(-1)

    if normalize:
        if norm_factor is None:
            values_for_factor = _off_diagonal_values(arr)
            positive = values_for_factor[values_for_factor > 0]
            if positive.size > 0:
                norm_factor = np.quantile(positive, quantile)
            else:
                norm_factor = np.max(np.abs(values_for_factor)) if values_for_factor.size > 0 else 1.0

        if norm_factor <= 0:
            norm_factor = 1.0

        arr = arr / norm_factor
        if clip:
            arr = np.clip(arr, 0, 1)

    if fill_diagonal and arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        np.fill_diagonal(arr, 1.0)

    return arr

# Funciones de estilo para figuras
#---------------------------------------------------------------------------
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # vendored src/figure_style.py
from figure_style import *

init_style()


# Matriz de presencia/ausencia de efectores
#---------------------------------------------------------------------------
def plot_effector_presence_matrix(matrix, cross_species_effectors, figures_output,
                                  dpi=600):
    """
    Plot the effector presence matrix with effectors as rows and strains as columns.
    """
    presence_color = "#2B1B88"
    bar_color = "#89BBFE"
    cross_species_effectors = set(cross_species_effectors)

    effector_order = matrix.sum(axis=1).sort_values(ascending=False).index
    strain_order = matrix.sum(axis=0).sort_values(ascending=False).index
    ordered_matrix = matrix.loc[effector_order, strain_order]

    fig = plt.figure(figsize=figsize(W_70*1.2, H_TALL*2), facecolor="white")
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[16, 3],
        wspace=0.05,
    )

    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_heatmap)

    cmap = mcolors.ListedColormap(["white", presence_color])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax_heatmap.imshow(
        ordered_matrix.values.astype(np.uint8),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        rasterized=False
    )

    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks(np.arange(ordered_matrix.shape[0]))
    ax_heatmap.set_yticklabels([
        (
            "*SopA/NleL-2" if effector == "SopA Shigella/NleL-2"
            else "*IpaH" if effector == "IpaH?"
            else "*NleL" if effector == "NleL-like?"
            else f"*{effector}" if effector in cross_species_effectors
            else effector
        )
        for effector in ordered_matrix.index
    ])
    style(ax_heatmap, xlabel="Strains", ylabel="Effectors")
    ax_heatmap.xaxis.set_label_position("top")
    ax_heatmap.tick_params(axis="x", length=0)
    ax_heatmap.tick_params(axis="y", length=0)
    for tick, effector in zip(ax_heatmap.get_yticklabels(), ordered_matrix.index):
        if effector in cross_species_effectors:
            tick.set_fontweight("bold")
    ax_heatmap.set_axisbelow(True)
    for spine_name, spine in ax_heatmap.spines.items():
        spine.set_visible(spine_name in ["top", "left", "bottom", "right"])
        spine.set_color("black")
        spine.set_linewidth(SPINE_LW)

    ax_right.barh(
        np.arange(ordered_matrix.shape[0]),
        ordered_matrix.sum(axis=1).values / ordered_matrix.shape[1] * 100,
        height=0.8,
        color=bar_color,
    )
    ax_right.set_xlabel("Prevalence")
    ax_right.xaxis.set_label_position("top")
    ax_right.set_xlim(0, 100)
    ax_right.set_xticks([50, 99])
    ax_right.set_xticklabels(["50%", "99%"])
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=TICK*1.1, length=2, pad=1)
    ax_right.grid(
        True,
        axis="x",
        linestyle="-",
        linewidth=GRID_LW,
        color="#CCCCCC",
        alpha=GRID_ALPHA,
        zorder=0
    )
    ax_right.set_axisbelow(True)
    for spine_name, spine in ax_right.spines.items():
        spine.set_visible(spine_name in ["top", "left"])
        spine.set_color("black")
        spine.set_linewidth(SPINE_LW)

    save(fig, f"{figures_output}/effector_presence_matrix",
        fmt = 'pdf',
        dpi=600)
    plt.show()

def plot_edge_weight_distribution_analysis(final_matrix, figures_output,
                                           dpi=400, bins=50):
    """
    Analyze and plot the absolute edge weight distribution for a final network.
    """
    edge_values = final_matrix.values[np.triu_indices(len(final_matrix), k=1)]
    abs_edge_values = np.abs(edge_values)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(abs_edge_values, percentiles)
    q1_abs_weight = float(np.percentile(abs_edge_values, 75))

    print("\n" + "=" * 60)
    print("  EDGE WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"  Total number of edges : {len(abs_edge_values)}")
    print(f"  Min abs weight        : {np.min(abs_edge_values):.6f}")
    print(f"  Max abs weight        : {np.max(abs_edge_values):.6f}")
    print(f"  Mean abs weight       : {np.mean(abs_edge_values):.6f}")
    print(f"  Median abs weight     : {np.median(abs_edge_values):.6f}")
    print(f"  Std abs weight        : {np.std(abs_edge_values):.6f}")
    print("\n  Percentile thresholds:")

    percentile_summary = {}
    for pct, value in zip(percentiles, percentile_values):
        n_above = int(np.sum(abs_edge_values >= value))
        pct_above = n_above / len(abs_edge_values) * 100
        percentile_summary[pct] = {
            "threshold": float(value),
            "n_above": n_above,
            "pct_above": float(pct_above),
        }
        print(f"  ├── P{pct:>2} = {value:.6f} | edges above: {n_above} ({pct_above:.2f}%)")

    print(f"\n  75th percentile threshold (Q1 absolute weights): {q1_abs_weight:.6f}")

    fig, ax = plt.subplots(figsize=figsize(W_70, H_STD))
    ax.hist(abs_edge_values, bins=bins, color="#89BBFE", edgecolor="black", linewidth=0.4)
    ax.axvline(q1_abs_weight, color="red", linestyle="--", linewidth=1.0)
    ax.text(
        q1_abs_weight,
        ax.get_ylim()[1] * 0.9,
        f"Q1 = {q1_abs_weight:.3f}",
        color="red",
        rotation=90,
        va="top",
        ha="right"
    )
    style(
        ax,
        xlabel="Absolute co-occurrence score",
        ylabel="Frequency",
        title="Distribution of absolute edge weights — Bootstrap Mean",
    )
    fig.tight_layout()
    fig.savefig(
        f"{figures_output}/bootstrap_mean_abs_edge_weights_histogram.png",
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)

    p60_abs_weight = float(np.percentile(abs_edge_values, 60))
    fig, ax = plt.subplots(figsize=figsize(W_70, H_STD))
    ax.hist(abs_edge_values, bins=bins, color="#89BBFE", edgecolor="black", linewidth=0.4)
    ax.axvline(q1_abs_weight, color="red", linestyle="--", linewidth=1.0)
    ax.text(
        q1_abs_weight,
        ax.get_ylim()[1] * 0.9,
        f"Q1 = {q1_abs_weight:.3f}",
        color="red",
        rotation=90,
        va="top",
        ha="right"
    )
    ax.set_xlim(p60_abs_weight, np.max(abs_edge_values))
    style(
        ax,
        xlabel="Absolute co-occurrence score",
        ylabel="Frequency",
        title="Distribution of absolute edge weights — Bootstrap Mean",
    )
    fig.tight_layout()
    fig.savefig(
        f"{figures_output}/bootstrap_mean_abs_edge_weights_histogram_top40.png",
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "n_edges": len(abs_edge_values),
        "min_abs_weight": float(np.min(abs_edge_values)),
        "max_abs_weight": float(np.max(abs_edge_values)),
        "mean_abs_weight": float(np.mean(abs_edge_values)),
        "median_abs_weight": float(np.median(abs_edge_values)),
        "std_abs_weight": float(np.std(abs_edge_values)),
        "percentiles": percentile_summary,
        "q1_abs_weight": q1_abs_weight,
    }

# Calculo co-occurrence con Phi y correccion por Fisher
#---------------------------------------------------------------------------
def cooccurrence_phi_fisher_rows(matrix, phi_threshold=0.15, fdr_alpha=0.05,
                                 normalize_adj=True, normalization_quantile=0.99):
    """
    Implementa el pipeline para matrices donde:
    - FILAS = efectores
    - COLUMNAS = cepas
    - valores = 0/1

    1. Calcula Phi coefficient para cada par de filas (efectores)
    2. Fisher exact test para corregir por prevalencia
    3. BH FDR correction
    4. Filtra por significancia + tamaño de efecto
    5. Devuelve matriz de similitud estilo sim_df
    """

    effectors = matrix.index
    n = len(effectors)

    # Matrices de salida
    phi_mat = np.zeros((n, n))
    pval_mat = np.ones((n, n))

    # Guardamos p-values para corregir al final
    pvals_list = []
    pairs = []

    # Recorremos pares de efectores (filas)
    for i in range(n):
        for j in range(i+1, n):

            A = matrix.iloc[i, :]   # fila i
            B = matrix.iloc[j, :]   # fila j

            # Construir tabla 2x2
            a = int(((A == 1) & (B == 1)).sum())
            b = int(((A == 1) & (B == 0)).sum())
            c = int(((A == 0) & (B == 1)).sum())
            d = int(((A == 0) & (B == 0)).sum())

            # Calcular phi
            denom = math.sqrt((a+b)*(c+d)*(a+c)*(b+d))
            phi = (a*d - b*c)/denom if denom > 0 else 0.0

            # Test exacto de Fisher
            _, p = fisher_exact([[a,b],[c,d]], alternative="two-sided")

            # Guardar valores
            phi_mat[i, j] = phi
            phi_mat[j, i] = phi

            pvals_list.append(p)
            pairs.append((i, j))

    # Corrección FDR de todos los p-values
    _, pvals_adj, _, _ = multipletests(pvals_list, method="fdr_bh")

    # Insertar p-values corregidos en matriz
    for k, (i, j) in enumerate(pairs):
        pval_mat[i, j] = pvals_adj[k]
        pval_mat[j, i] = pvals_adj[k]

    # Matriz final de similitudes
    sim = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                sim[i, j] = 1.0
            else:
                if (abs(phi_mat[i, j]) >= phi_threshold) and (pval_mat[i, j] <= fdr_alpha):
                    sim[i, j] = phi_mat[i, j]
                else:
                    sim[i, j] = 0.0

    sim = normalize_similarity_matrix(
        sim,
        normalize=normalize_adj,
        quantile=normalization_quantile,
        clip=True,
        fill_diagonal=True,
    )
    sim_df = pd.DataFrame(sim, index=effectors, columns=effectors)
    
    return sim_df, phi_mat, pval_mat

def cooccurrence_jaccard_rows(matrix, normalize_adj=True, normalization_quantile=0.99):
    """
    Compute Jaccard similarity between rows (effectors) with the same
    normalisation interface as the other inferred matrices.
    """
    dist = pdist(matrix.values, metric='jaccard')
    sim_arr = 1 - squareform(dist)
    np.nan_to_num(sim_arr, copy=False, nan=0.0)
    sim_arr = normalize_similarity_matrix(
        sim_arr,
        normalize=normalize_adj,
        quantile=normalization_quantile,
        clip=True,
        fill_diagonal=True,
    )
    return pd.DataFrame(sim_arr, index=matrix.index, columns=matrix.index)

# Heatmap co-occurrence
#---------------------------------------------------------------------------
def heatmap_coocurrence(sim_df, title, figures_output, dpi=600):
    # Crear máscara para ocultar la mitad superior y diagonal
    mask = np.triu(np.ones_like(sim_df, dtype=bool))
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(sim_df, mask=mask, cmap="Blues", annot=False)
    style(ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right')
    ax.set_yticklabels(ax.get_yticklabels())
    # Mostrar solo cada 2 etiquetas en x
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 2 != 0:
            label.set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{figures_output}/effector_similarity_heatmap.png", dpi=dpi)
    plt.show()

# Exportar grafo de co-ocurrencia + prevalencia
#---------------------------------------------------------------------------
# OPCION 1 - Seleccionar por threshold
def export_grafo_json(sim_df, effector_count, total, data_output, threshold=0.0):
    """
    Exporta un grafo en formato JSON para Cytoscape:
    - Nodos: efectores con atributo 'prevalence'
    - Edges: pares con similitud Jaccard > threshold
    """

    # Crear lista de nodos con prevalencia
    nodes = []
    for effector in sim_df.index:
        prevalence = int(effector_count.loc[effector]) if effector in effector_count.index else 0
        nodes.append({
            "data": {
                "id": effector,
                "label": effector,
                "prevalence": prevalence/total
            }
        })


    # Crear lista de edges (filtrando por umbral)
    edges = []
    for i in range(len(sim_df)):
        for j in range(i+1, len(sim_df)):
            weight = sim_df.iloc[i, j]
            if weight > threshold:
                edges.append({
                    "data": {
                        "source": sim_df.index[i],
                        "target": sim_df.columns[j],
                        "weight": round(weight, 4),
                        "weight_binary": "True"
                    }})
            else:
                edges.append({
                    "data": {
                        "source": sim_df.index[i],
                        "target": sim_df.columns[j],
                        "weight": round(weight, 4),
                        "weight_binary": "False"
                    }})
    print(len(edges), ' interacciones en el grafo')
    # Construir estructura JSON
    graph_json = {
        "elements": {
            "nodes": nodes,
            "edges": edges
        }
    }

    # Guardar en archivo
    output_path = f"{data_output}\\effectornetwork_coocurrence_prevalence.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, indent=4)

    print(f"Grafo exportado en formato JSON para Cytoscape: {output_path}")

# OPCION 2 - Seleccionar topk
def export_grafo_json_topk(sim_df, effector_count, total, data_output,
                           top_percent=25, filename="Graph",
                           use_absolute_weights=False):
    """
    Igual que la original, pero el JSON final se exporta en formato:
    {
        "elements": {
            "nodes": [...],
            "edges": [...]
        }
    }
    con los campos label y prevalence en nodos,
    y weight_binary="True" en las aristas.
    """

    import numpy as np
    import pandas as pd
    import json
    import os

    print("\n[export_grafo_json_topk] Iniciando exportación...")

    sim_df = sim_df.astype(float)

    # ------------------------------
    # 2) Pesos no-nulos (solo triángulo superior, excluye diagonal)
    # ------------------------------
    _n = len(sim_df)
    _tri_i, _tri_j = np.triu_indices(_n, k=1)
    tri_vals = sim_df.values[_tri_i, _tri_j]
    ranking_vals = np.abs(tri_vals) if use_absolute_weights else tri_vals
    nonzero = ranking_vals[ranking_vals > 0]

    if len(nonzero) == 0:
        raise ValueError("No hay pesos no nulos.")

    # ------------------------------
    # 3) Threshold
    # ------------------------------
    k = int(len(nonzero) * (top_percent / 100))
    k = max(k, 1)

    sorted_nonzero = np.sort(nonzero)
    threshold = sorted_nonzero[-k]

    ranking_label = "|weight|" if use_absolute_weights else "weight"
    print(f"[threshold] top {top_percent}% by {ranking_label} -> {threshold:.6f}")

    # ------------------------------
    # 5) effector_count a dict
    # ------------------------------
    if isinstance(effector_count, pd.Series):
        effector_count = effector_count.to_dict()

    # ------------------------------
    # 6) Lista de nodos (nuevo formato)
    # ------------------------------
    nodes = []
    for effector in sim_df.index:
        prevalence = effector_count[effector] / total

        nodes.append({
            "data": {
                "id": effector,
                "label": effector,
                "prevalence": float(prevalence)
            }
        })

    # ------------------------------
    # 7) Aristas (nuevo formato)
    # ------------------------------
    edges = []
    for i, source in enumerate(sim_df.index):
        for j, target in enumerate(sim_df.columns):

            if i >= j:
                continue  # <-- solo triángulo superior, evita duplicados y self-loops

            weight = sim_df.iat[i, j]

            rank_weight = abs(weight) if use_absolute_weights else weight

            if rank_weight >= threshold and rank_weight > 0:
                edges.append({
                    "data": {
                        "source": source,
                        "target": target,
                        "weight": float(weight),
                        "abs_weight": float(abs(weight)),
                        "weight_sign": "negative" if weight < 0 else "positive",
                        "weight_binary": "True"
                    }
                })


    print(f"[exportación] Total edges exportados: {len(edges)}")

    # ------------------------------
    # 8) Empaquetar JSON en formato requerido
    # ------------------------------
    graph_json = {
        "elements": {
            "nodes": nodes,
            "edges": edges
        }
    }

    # ------------------------------
    # 9) Guardar archivo
    # ------------------------------
    os.makedirs(data_output, exist_ok=True)
    output_path = os.path.join(data_output, f"effectornetwork_top{top_percent}_{filename}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, indent=4)

    print(f"[OK] Grafo exportado en: {output_path}\n")

# OPCION 3 - Soft weighting
def export_grafo_json_softweighting(sim_df, effector_count, total, data_output, 
                                    sigma_min=0.02, rescale=True, filename_suffix="softw"):
    """
    Exporta un grafo JSON para Cytoscape usando soft-weighting:
        w* = max(0, w - sigma_min)
    y opcionalmente reescalado a [0,1].

    - sim_df: matriz NxN con weights en [0,1]
    - sigma_min: mínimo umbral suave (0.02 recomendado)
    - rescale: si True, reescala tras suavizar
    """

    import numpy as np
    import json
    import os

    # ---------- SOFT WEIGHTING ----------
    S = sim_df.values.astype(float)

    # aplicar suavizado
    S_sw = np.clip(S - sigma_min, 0, None)

    # reescalar si se desea (opcional)
    if rescale:
        max_val = S_sw.max()
        if max_val > 0:
            S_sw = S_sw / max_val

    # dataframe final para facilidad
    S_df = pd.DataFrame(S_sw, index=sim_df.index, columns=sim_df.columns)

    # ---------- NODOS ----------
    nodes = []
    for effector in S_df.index:
        prevalence = int(effector_count.loc[effector]) if effector in effector_count.index else 0
        nodes.append({
            "data": {
                "id": effector,
                "label": effector,
                "prevalence": prevalence / total
            }
        })

    # ---------- ARISTAS ----------
    edges = []
    N = len(S_df)
    for i in range(N):
        for j in range(i+1, N):
            w = S_df.iloc[i, j]
            if w > 0:  # mantener SOLO las aristas suavizadas
                edges.append({
                    "data": {
                        "source": S_df.index[i],
                        "target": S_df.columns[j],
                        "weight": round(float(w), 4),
                        "weight_binary": "True" if w > 0 else "False"
                    }
                })

    print(f"{len(edges)} interacciones con soft-weighting (sigma_min={sigma_min})")

    # ---------- JSON ----------
    graph_json = {
        "elements": {
            "nodes": nodes,
            "edges": edges
        }
    }

    # ----------- GUARDAR ----------
    file = os.path.join(data_output, f"effectornetwork_{filename_suffix}.json")
    with open(file, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, indent=4)

    print(f"Grafo exportado: {file}")

    return S_df, graph_json

# Regresión logística para co-ocurrencia
#---------------------------------------------------------------------------
def neighborhood_selection_logistic(matrix, 
                                    Cs = [0.1, 1, 10], 
                                    cv=5, 
                                    penalty='l1', 
                                    solver='saga',
                                    max_iter=5000,
                                    tol=1e-4,
                                    symm_rule='average',   # 'average' or 'and' or 'or'
                                    zero_tol = 1e-6,
                                    normalize_adj=True,
                                    normalization_quantile=0.99):
    """
    Neighborhood selection para datos binarios (filas = efectores, columnas = cepas).
    Devuelve:
      - adj_df: DataFrame (efectores x efectores) con pesos simétricos (float >=0)
      - coef_matrix: matriz (n_effectors x n_effectors) con coeficientes (no simetrizada)
    Parámetros:
      - Cs: lista de valores C (inverso de regularización) para CV (mayor C -> menos regularización)
      - symm_rule: forma de simetrizar ('average' toma media de |coef_ij| y |coef_ji|;
                                      'average_signed' toma media de coef_ij y coef_ji
                                          conservando el signo (adj puede ser negativa);
                                      'and' pone 1 si ambos direccionales no-cero;
                                      'or' pone 1 si cualquiera es no-cero)
      - zero_tol: umbral numérico para considerar coef = 0 (debido a numerics)
    """

    # matrix: filas=efectores, columnas=cepas
    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)

    effectors = matrix.index.tolist()
    n_effectors = len(effectors)
    X = matrix.values.T  # ahora rows = samples(cepas), cols = variables(efectores)
    n_samples, n_features = X.shape

    # Standardizar predictors (no estrictamente necesario para L1-logistic, pero mejora CV)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    # Matriz de coeficientes (directional): coef_matrix[j, k] = coef predicting j using k (k index in predictors)
    coef_matrix = np.zeros((n_features, n_features))  # row = target j, col = predictor k

    # warnings silencing for convergence (pero no ocultar errores)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Loop: para cada variable j ajusta modelo predictivo
    for j in range(n_features):
        y = X[:, j].astype(int)
        # Predictores: todas menos j
        mask = np.ones(n_features, dtype=bool)
        mask[j] = False
        X_pred = Xs[:, mask]

        # Si y es constante (todo 0 o todo 1) no se puede ajustar; dejar coef = 0
        if y.sum() == 0 or y.sum() == len(y):
            # deja la fila de coeficientes a 0s
            continue

        # LogisticRegressionCV con L1 regularization (saga)
        # usa scoring 'roc_auc' si hay variación; CV automático busca mejor C
        try:
            lr_cv = LogisticRegressionCV(Cs=Cs, cv=cv, penalty=penalty, solver=solver,
                                         scoring='roc_auc', max_iter=max_iter, tol=tol,
                                         fit_intercept=True, class_weight='balanced', n_jobs=1)
            lr_cv.fit(X_pred, y)
            coefs = lr_cv.coef_.ravel()  # longitud n_features-1
        except Exception as e:
            # fallback: un ajuste sin CV con C=1
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1.0, penalty=penalty, solver=solver, max_iter=max_iter, tol=tol)
            lr.fit(X_pred, y)
            coefs = lr.coef_.ravel()

        # Volvemos a alinear en la matriz completa (poniendo 0 en la diagonal posición j)
        full_coefs = np.zeros(n_features)
        full_coefs[mask] = coefs
        coef_matrix[j, :] = full_coefs

    # Simetrizar: construir matriz de pesos no-negativos
    adj = np.zeros_like(coef_matrix)

    if symm_rule == 'average':
        # adj_ij = (|coef_ij| + |coef_ji|) / 2
        adj = (np.abs(coef_matrix) + np.abs(coef_matrix.T)) / 2.0
    elif symm_rule == 'average_signed':
        # adj_ij = (coef_ij + coef_ji) / 2  -> conserva el signo del coeficiente
        # (necesario para que L1 sea comparable con VAE/Bootstrap/Phi, que sí tienen signo)
        adj = (coef_matrix + coef_matrix.T) / 2.0
    elif symm_rule == 'and':
        # adj_ij = 1 if both directions non-zero (could be weighted by mean abs)
        both_nonzero = (np.abs(coef_matrix) > zero_tol) & (np.abs(coef_matrix.T) > zero_tol)
        adj = both_nonzero.astype(float)
    elif symm_rule == 'or':
        any_nonzero = (np.abs(coef_matrix) > zero_tol) | (np.abs(coef_matrix.T) > zero_tol)
        adj = any_nonzero.astype(float)
    else:
        raise ValueError("symm_rule must be 'average', 'average_signed', 'and' or 'or'")

    adj = normalize_similarity_matrix(
        adj,
        normalize=normalize_adj,
        quantile=normalization_quantile,
        clip=True,
        fill_diagonal=True,
    )

    adj_df = pd.DataFrame(adj, index=effectors, columns=effectors)
    coef_df = pd.DataFrame(coef_matrix, index=effectors, columns=effectors)

    return adj_df, coef_df


# Bagging para Phi y L1 (mismo mecanismo que bootstrap_vae_network)
# -------------------------------------------------------------------
# Las réplicas internas de bagging son independientes entre sí y se reparten
# con joblib.Parallel(n_jobs=...). El VAE (bernoulli_vae_network,
# bootstrap_vae_network, bootstrap_vae_statistical_network) se deja SIEMPRE en
# serie: no se paraleliza aquí para no competir por recursos con PyTorch.
def bagged_phi_network(matrix, n_bootstrap=75, sample_fraction=0.6, seed=42,
                       phi_threshold=0.15, fdr_alpha=0.05, n_jobs=-2):
    """
    Bagging del coeficiente Phi, análogo al bootstrap del VAE
    (bootstrap_vae_network), para que la Tabla 1 compare estabilidad de forma
    simétrica entre los cinco métodos.

    - En cada una de las `n_bootstrap` réplicas se remuestrean CEPAS (columnas)
      CON REEMPLAZO: idx = rng.choice(n_samples, size=int(sample_fraction*n_samples),
      replace=True) — idéntico a bootstrap_vae_network.
    - Se llama a cooccurrence_phi_fisher_rows y se guarda `phi_mat` (el SEGUNDO
      valor devuelto): la matriz Phi CRUDA n x n, NO `sim_df` (que ya viene
      filtrada por Fisher/BH). El VAE se bootstrapea con normalize_adj=False sobre
      valores crudos sin filtro por réplica, así que Phi sigue la misma lógica.
      Cualquier criterio de significancia se aplica DESPUÉS de promediar, nunca
      antes.
    - Se promedian las `n_bootstrap` matrices `phi_mat` elemento a elemento.

    Paralelización
    --------------
    Los `idx` de bootstrap se generan SECUENCIALMENTE desde np.random.default_rng(seed)
    (mismo orden que tendría el bucle en serie) y luego las réplicas se reparten
    con joblib.Parallel(n_jobs=n_jobs) (n_jobs=-2 -> todos los núcleos menos uno).
    cooccurrence_phi_fisher_rows es determinista (no usa RNG), así que el
    resultado es BIT-IDÉNTICO al del bucle en serie. n_jobs=1 ejecuta en serie
    en el mismo proceso (referencia de verificación).

    Devuelve
    --------
    consensus_df : pd.DataFrame (efectores x efectores)
        Media elemento a elemento de las réplicas de Phi crudo.
    replicates : list[pd.DataFrame]
        Las `n_bootstrap` matrices Phi crudas individuales.
    """
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    effectors = matrix.index
    n_samples = matrix.shape[1]
    size = int(sample_fraction * n_samples)

    # Orden idéntico al del bucle en serie: idx[b] es el que la iteración b
    # habría sacado del mismo Generator.
    idx_per_rep = [rng.choice(n_samples, size=size, replace=True)
                   for _ in range(n_bootstrap)]

    def _replica_phi(idx):
        _, phi_mat, _ = cooccurrence_phi_fisher_rows(
            matrix.iloc[:, idx],
            phi_threshold,
            fdr_alpha,
        )
        return np.asarray(phi_mat, dtype=float)

    mats = Parallel(n_jobs=n_jobs)(delayed(_replica_phi)(idx) for idx in idx_per_rep)
    replicates = [pd.DataFrame(m, index=effectors, columns=effectors) for m in mats]

    stack = np.stack([r.values for r in replicates], axis=0)
    consensus_df = pd.DataFrame(stack.mean(axis=0), index=effectors, columns=effectors)
    return consensus_df, replicates


def bagged_l1_network(matrix, n_bootstrap=50, sample_fraction=0.6, seed=42,
                      n_jobs=-2, **kwargs):
    """
    Bagging de la regresión logística L1 (neighborhood_selection_logistic),
    con el mismo esquema de remuestreo que bootstrap_vae_network / bagged_phi_network:
    CEPAS CON REEMPLAZO, sample_fraction=0.6, `n_bootstrap` réplicas.

    - En cada réplica se llama a neighborhood_selection_logistic(matrix.iloc[:, idx],
      normalize_adj=False, symm_rule='average_signed') y se guarda `adj_df`
      (matriz simétrica CON signo, sin normalizar ni filtrar por réplica).
    - Se promedian las `n_bootstrap` matrices elemento a elemento.
    - `kwargs` extra se reenvían a neighborhood_selection_logistic.

    Nota sobre n_bootstrap
    ----------------------
    Por defecto 50 (se bajó de 75; decidido tras medir el coste de 'saga').
    bagged_phi_network sigue en 75.

    Nota sobre el solver
    -------------------
    Usa el solver por defecto de neighborhood_selection_logistic ('saga'). La
    firma de neighborhood_selection_logistic NO se toca. (Se evaluó cambiar solo
    aquí a 'liblinear' para acelerar el bagging, pero la comparación de 25
    réplicas no alcanzó los umbrales de acuerdo fijados de antemano —
    Pearson medio 0.865 < 0.90, acuerdo de signo 78.8% < 90% — así que se
    mantiene 'saga'. Ver logs/cmp_saga_liblinear_25rep.out.)

    Reproducibilidad y paralelización
    ---------------------------------
    LogisticRegressionCV/saga con random_state=None (así lo llama
    neighborhood_selection_logistic, que NO se toca) lee el RNG GLOBAL de NumPy
    para el orden de muestreo del SGD -> sin fijarlo, el resultado por réplica
    NO es reproducible. Aquí se fija np.random.seed(seed + b) al principio del
    trabajo de cada réplica b (semilla derivada de forma reproducible del `seed`
    de la función y del índice de réplica -> cada número es trazable).

    Con esa semilla-por-réplica ya determinista, la versión paralela
    (joblib.Parallel(n_jobs=n_jobs), n_jobs=-2 = todos los núcleos menos uno) es
    BIT-IDÉNTICA a la versión en serie (n_jobs=1) con esa MISMA semilla-por-
    réplica: cada worker fija np.random.seed(seed + b) antes de su fit, y el
    `idx` de cada réplica se pre-genera en serie desde np.random.default_rng(seed).

    Devuelve
    --------
    consensus_df : pd.DataFrame (efectores x efectores)
    replicates : list[pd.DataFrame]
    """
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    effectors = matrix.index
    n_samples = matrix.shape[1]
    size = int(sample_fraction * n_samples)

    # idx[b] = el que la iteración b del bucle en serie sacaría del mismo
    # Generator (np.random.default_rng(seed)); no lo afecta np.random.seed().
    idx_per_rep = [rng.choice(n_samples, size=size, replace=True)
                   for _ in range(n_bootstrap)]

    def _replica_l1(b, idx):
        # Semilla del RNG GLOBAL de NumPy, determinista y trazable: seed + b.
        # saga (dentro de neighborhood_selection_logistic) la lee para el orden
        # de muestreo del SGD.
        np.random.seed(seed + b)
        adj_df, _ = neighborhood_selection_logistic(
            matrix.iloc[:, idx],
            normalize_adj=False,
            symm_rule='average_signed',
            **kwargs,
        )
        return adj_df.values

    mats = Parallel(n_jobs=n_jobs)(
        delayed(_replica_l1)(b, idx) for b, idx in enumerate(idx_per_rep)
    )
    replicates = [pd.DataFrame(m, index=effectors, columns=effectors) for m in mats]

    stack = np.stack([r.values for r in replicates], axis=0)
    consensus_df = pd.DataFrame(stack.mean(axis=0), index=effectors, columns=effectors)
    return consensus_df, replicates


# Bernouilli-VAE para co-ocurrencia
#---------------------------------------------------------------------------
def bernoulli_vae_network(
    matrix,
    hidden_dim = 32,
    latent_dim = 8,
    epochs=400,
    batch_size=32,
    lr=1e-3,
    beta_kl=0.1,
    device='cpu',
    seed=42,
    verbose=True,
    normalize_adj=True,
    normalization_quantile=0.99,
    symm_rule='average',
    must_link=None,
    gamma_max=1e-3,      # fuerza máxima regularización
    burn_in_frac=0.3,    # fracción inicial sin regularización
    ramp_frac=0.3        # fracción de suavizado
):
    """
    Bernoulli-VAE con regularización must-link geométrica:
        R = gamma(t) * sum( || W_i - W_j ||^2 )

    donde gamma(t) aumenta en fases (burn-in -> ramp -> full).
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import pandas as pd
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics.pairwise import cosine_similarity

    torch.manual_seed(seed)
    np.random.seed(seed)

    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)

    effectors = matrix.index.tolist()
    n_features = len(effectors)

    # X: (n_samples x n_features)
    X = matrix.values.T.astype(np.float32)
    n_samples = X.shape[0]

    device = torch.device(device)

    tensor_X = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_X), batch_size=batch_size, shuffle=True)

    # -----------------------------
    # Encoder / Decoder
    # -----------------------------
    class Encoder(nn.Module):
        def __init__(self, n_features, hidden_dim, latent_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.1)
            )
            self.mu = nn.Linear(hidden_dim, latent_dim)
            self.logvar = nn.Linear(hidden_dim, latent_dim)
        def forward(self, x):
            h = self.net(x)
            return self.mu(h), self.logvar(h)

    class DecoderLinear(nn.Module):
        def __init__(self, latent_dim, n_features):
            super().__init__()
            self.linear = nn.Linear(latent_dim, n_features, bias=True)
        def forward(self, z):
            return self.linear(z)  # logits

    encoder = Encoder(n_features, hidden_dim, latent_dim).to(device)
    decoder = DecoderLinear(latent_dim, n_features).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    # -----------------------------
    # Must-link index mapping
    # -----------------------------
    must_link_idx = []
    if must_link:
        for a, b in must_link:
            if a in effectors and b in effectors:
                must_link_idx.append((effectors.index(a), effectors.index(b)))
            else:
                print(f"[WARNING] must-link ignorado: ({a},{b})")

    # -----------------------------
    # Must-link schedule
    # -----------------------------
    burn_in_epochs = int(burn_in_frac * epochs)
    ramp_epochs = int(ramp_frac * epochs)

    def gamma_t(epoch):
        """Gamma aumenta desde 0 hasta gamma_max."""
        if epoch < burn_in_epochs:
            return 0.0
        elif epoch < burn_in_epochs + ramp_epochs:
            alpha = (epoch - burn_in_epochs) / ramp_epochs
            return alpha * gamma_max
        else:
            return gamma_max

    history = []

    # =======================================================
    # TRAINING LOOP
    # =======================================================
    for epoch in range(epochs):

        encoder.train()
        decoder.train()

        running_loss = 0.0

        # -----------------------------
        # Standard VAE training
        # -----------------------------
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            mu, logvar = encoder(batch_x)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            logits = decoder(z)

            recon_loss = bce_loss(logits, batch_x)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + beta_kl * kl
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # -----------------------------
        # Must-link geometric regularization
        # -----------------------------
        gamma = gamma_t(epoch)

        if must_link_idx and gamma > 0:
            W = decoder.linear.weight  # (features x latent_dim)

            reg = 0.0
            for i, j in must_link_idx:
                reg += torch.sum((W[i] - W[j])**2)

            optimizer.zero_grad()
            (gamma * reg).backward()
            optimizer.step()

        avg_loss = running_loss / n_samples
        history.append(avg_loss)

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            if must_link_idx:
                with torch.no_grad():
                    Wv = decoder.linear.weight
                    ml_distances = [torch.norm(Wv[i]-Wv[j]).item() for i,j in must_link_idx]
            else:
                ml_distances = []

            print(
                f"Epoch {epoch+1}/{epochs} "
                f"- loss/sample {avg_loss:.4f} "
                f"| gamma={gamma:.6f} "
                + (f"| ML mean dist={np.mean(ml_distances):.4f}" if ml_distances else "")
            )

    # =======================================================
    # BUILD FINAL ADJ USING COSINE SIMILARITY
    # =======================================================
    W = decoder.linear.weight.detach().cpu().numpy()
    S = cosine_similarity(W)

    S = normalize_similarity_matrix(
        S,
        normalize=normalize_adj,
        quantile=normalization_quantile,
        clip=True,
        fill_diagonal=False,
    )

    # simmetrization
    if symm_rule == 'average':
        adj = 0.5 * (S + S.T)
    elif symm_rule == 'or':
        adj = ((S > 0) | (S.T > 0)).astype(float) * S
    elif symm_rule == 'and':
        adj = ((S > 0) & (S.T > 0)).astype(float) * S
    else:
        raise ValueError("symm_rule must be 'average','or','and'")

    np.fill_diagonal(adj, 1.0)
    adj_df = pd.DataFrame(adj, index=effectors, columns=effectors)

    latent_cols = [f"z{k}" for k in range(latent_dim)]
    coef_df = pd.DataFrame(W, index=effectors, columns=latent_cols)

    extras = {
        'model': {'encoder': encoder, 'decoder': decoder},
        'history': history,
        'W': W
    }

    return adj_df, coef_df, extras

def bootstrap_vae_network(
    matrix,
    n_bootstrap=50,
    sample_fraction=0.8,
    must_link=None,
    vae_params=None,
    device='cpu',
    seed=42,
    verbose=True,
    normalize_for_stats=True,
    normalization_quantile=0.99,
    topk_for_stability=50,
    sd_tol=1e-3
):
    """
    Bootstrap sobre Bernoulli-VAE para evaluar robustez del estimador.
    Devuelve:
      - agg_adj_df: DataFrame (N x N) con la matriz agregada (promedio) normalizada [0,1]
      - S_boot: numpy array (B, N, N) con las B réplicas normalizadas en la misma escala
      - stats: dict con métricas de estabilidad (mean_stability, median_stability, frac_sd_small,
               topk_jaccard_mean, pairwise_corr_mean, per_edge_mu, per_edge_sd, ...)
    Parámetros:
      - matrix: pd.DataFrame filas=efectores, cols=cepas
      - n_bootstrap, sample_fraction, must_link, vae_params como antes
      - normalize_for_stats: si True normaliza todas las S por el máximo global (recomendado)
      - topk_for_stability: k para Jaccard top-k stability
      - sd_tol: umbral para considerar sd "pequeña"
    """
    import numpy as np
    import pandas as pd
    from itertools import combinations
    from scipy.stats import pearsonr, spearmanr

    rng = np.random.default_rng(seed)

    if vae_params is None:
        vae_params = {}

    # Estructura
    effectors = matrix.index.tolist()
    n_effectors = len(effectors)
    n_samples = matrix.shape[1]

    # Guardar S crudas (sin normalizar internamente en bernoulli_vae_network)
    S_raw = np.zeros((n_bootstrap, n_effectors, n_effectors), dtype=float)

    # Loop de bootstrap: submuestreo de cepas y entrenamiento del VAE
    for b in range(n_bootstrap):
        if verbose and (b % max(1, n_bootstrap//10) == 0):
            print(f"[bootstrap] {b+1}/{n_bootstrap}")

        idx = rng.choice(n_samples, size=int(sample_fraction * n_samples), replace=True)
        boot_matrix = matrix.iloc[:, idx]

        # llamar al VAE pidiendo que NO normalice internamente (normalize_adj=False)
        adj_df_b, coef_b, extras_b = bernoulli_vae_network(
            boot_matrix,
            **vae_params,
            must_link=must_link,
            device=device,
            verbose=False,
            normalize_adj=False   # IMPORTANTE: guardamos S cruda por bootstrap
        )

        S_raw[b, :, :] = adj_df_b.values

    # También calcular el modelo completo (opcional): S_full_raw
    S_full_df_raw, coef_full, extras_full = bernoulli_vae_network(
        matrix,
        **vae_params,
        must_link=must_link,
        device=device,
        verbose=False,
        normalize_adj=False
    )
    S_full_raw = S_full_df_raw.values

    # ---- Normalización por cuartiles (para que todas las réplicas sean comparables) ----
    if normalize_for_stats:
        # escalar por un cuantil alto para evitar outliers en el bootstrap
        # cuantil calculado sobre TODAS las réplicas + modelo completo
        offdiag_mask = ~np.eye(n_effectors, dtype=bool)
        all_values = np.concatenate([
            S_raw[:, offdiag_mask].reshape(-1),
            S_full_raw[offdiag_mask].reshape(-1)
        ])
        positive = all_values[all_values > 0]
        norm_factor = np.quantile(positive, normalization_quantile) if positive.size > 0 else 1.0
        S_boot = normalize_similarity_matrix(
            S_raw,
            normalize=True,
            quantile=normalization_quantile,
            clip=True,
            fill_diagonal=False,
            norm_factor=norm_factor,
        )
        S_full = normalize_similarity_matrix(
            S_full_raw,
            normalize=True,
            quantile=normalization_quantile,
            clip=True,
            fill_diagonal=True,
            norm_factor=norm_factor,
        )
    else:
        S_boot = S_raw.copy()
        S_full = S_full_raw.copy()
    S_boot_norm = S_boot

    # ---- Matriz agregada (promedio) ----
    agg_adj = S_boot.mean(axis=0)
    # opcional simmetrizar y fijar diagonal 1
    agg_adj = 0.5 * (agg_adj + agg_adj.T)
    np.fill_diagonal(agg_adj, 1.0)

    agg_adj_df = pd.DataFrame(agg_adj, index=effectors, columns=effectors)
    S_full_df = pd.DataFrame(S_full, index=effectors, columns=effectors)
    matrices_individuales = [
        pd.DataFrame(S_boot[b], index=effectors, columns=effectors)
        for b in range(n_bootstrap)
    ]
    consensus_matrix = agg_adj_df

    # ---- Métricas por arista (upper triangular) ----
    tri = np.triu_indices(n_effectors, k=1)
    B = n_bootstrap
    M = len(tri[0])

    # convertir a vectores (B x M)
    S_vecs = np.zeros((B, M), dtype=float)
    for b in range(B):
        S_vecs[b, :] = S_boot[b][tri]

    mu = S_vecs.mean(axis=0)        # (M,)
    sd = S_vecs.std(axis=0)         # (M,)
    stability_ratio = mu / (sd + 1e-12)

    mean_stability = float(np.mean(stability_ratio))
    median_stability = float(np.median(stability_ratio))
    frac_sd_small = float(np.mean(sd < sd_tol))

    # ---- Top-k Jaccard stability ----
    def topk_sets_from_S(Sarr, k):
        sets = []
        for b in range(Sarr.shape[0]):
            vals = Sarr[b][tri]
            idx_sorted = np.argsort(vals)[::-1][:k]
            sets.append(set(idx_sorted))
        return sets

    topk_sets = topk_sets_from_S(S_boot, topk_for_stability)
    # pairwise Jaccard mean
    jaccs = []
    for i, j in combinations(range(B), 2):
        A = topk_sets[i]; Bset = topk_sets[j]
        inter = len(A & Bset)
        union = len(A | Bset)
        jaccs.append(inter/union if union > 0 else 0.0)
    topk_jaccard_mean = float(np.mean(jaccs)) if len(jaccs) > 0 else 0.0
    topk_jaccard_std = float(np.std(jaccs)) if len(jaccs) > 0 else 0.0

    # ---- Pairwise matrix correlations between réplicas ----
    pair_corrs = []
    for i, j in combinations(range(B), 2):
        xi = S_vecs[i]; xj = S_vecs[j]
        # Pearson
        try:
            r = pearsonr(xi, xj)[0]
        except Exception:
            r = 0.0
        pair_corrs.append(r)
    pairwise_corr_mean = float(np.mean(pair_corrs)) if len(pair_corrs) > 0 else 0.0
    pairwise_corr_std = float(np.std(pair_corrs)) if len(pair_corrs) > 0 else 0.0

    # ---- Packaging per-edge mu/sd back to matrices for debugging if needed ----
    mu_mat = np.zeros((n_effectors, n_effectors), dtype=float)
    sd_mat = np.zeros((n_effectors, n_effectors), dtype=float)
    mu_mat[tri] = mu
    mu_mat[(tri[1], tri[0])] = mu  # symmetric
    sd_mat[tri] = sd
    sd_mat[(tri[1], tri[0])] = sd

    stats = {
        'mean_stability_ratio': mean_stability,
        'median_stability_ratio': median_stability,
        'frac_sd_small': frac_sd_small,
        'topk_jaccard_mean': topk_jaccard_mean,
        'topk_jaccard_std': topk_jaccard_std,
        'pairwise_corr_mean': pairwise_corr_mean,
        'pairwise_corr_std': pairwise_corr_std,
        'per_edge_mu_vector': mu,   # length M
        'per_edge_sd_vector': sd,   # length M
        'per_edge_mu_matrix': mu_mat,
        'per_edge_sd_matrix': sd_mat,
        'agg_adj_df': agg_adj_df,
        'S_full_df': S_full_df
    }

    if verbose:
        print("[bootstrap_vae_network] estabilidad resumen:")
        print(f"  mean_stability_ratio: {mean_stability:.4f}")
        print(f"  median_stability_ratio: {median_stability:.4f}")
        print(f"  frac_sd < {sd_tol}: {frac_sd_small:.4f}")
        print(f"  top{topk_for_stability} jaccard mean: {topk_jaccard_mean:.4f}")
        print(f"  pairwise corr mean: {pairwise_corr_mean:.4f}")

    return consensus_matrix, S_full_df, S_boot_norm, matrices_individuales

def bootstrap_vae_statistical_network(matrix, 
                                     vae_params,
                                     n_bootstrap=50, 
                                     sample_fraction=0.8,
                                     must_link=None,
                                     device='cpu',
                                     seed=42,
                                     verbose=True,
                                     normalize_for_stats=True,
                                     normalization_quantile=0.99):
    """
    Versión corregida CON NORMALIZACIÓN POR CUANTIL 0.99 (idéntica al bootstrap promedio).
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)

    effectors = matrix.index.tolist()
    n_effectors = len(effectors)
    n_samples = matrix.shape[1]

    # almacenar S crudas (no normalizadas)
    S_boot = np.zeros((n_bootstrap, n_effectors, n_effectors), dtype=np.float32)

    # ---------------- BOOTSTRAP ----------------
    for b in range(n_bootstrap):
        if verbose and (b % max(1, n_bootstrap//10) == 0):
            print(f"[bootstrap] {b+1}/{n_bootstrap}")

        sample_idx = rng.choice(n_samples, size=int(n_samples*sample_fraction), replace=True)
        boot_matrix = matrix.iloc[:, sample_idx]

        adj_df_b, coef_b, extras_b = bernoulli_vae_network(
            boot_matrix,
            **vae_params,
            must_link=must_link,
            device=device,
            verbose=False,
            normalize_adj=False   # guardar S cruda
        )

        S_boot[b, :, :] = adj_df_b.values

    # ---------------- MODELO COMPLETO ----------------
    S_full_df_raw, coef_full, extras_full = bernoulli_vae_network(
        matrix,
        **vae_params,
        must_link=must_link,
        device=device,
        verbose=False,
        normalize_adj=False
    )
    S_full_raw = S_full_df_raw.values

    # ---------------- NORMALIZACIÓN POR CUANTIL 0.99 ----------------
    # idéntica a bootstrap_vae_network
    if normalize_for_stats:
        offdiag_mask = ~np.eye(n_effectors, dtype=bool)
        all_values = np.concatenate([
            S_boot[:, offdiag_mask].reshape(-1),
            S_full_raw[offdiag_mask].reshape(-1)
        ])
        positive = all_values[all_values > 0]
        norm_factor = np.quantile(positive, normalization_quantile) if positive.size > 0 else 1.0
        S_boot_norm = normalize_similarity_matrix(
            S_boot,
            normalize=True,
            quantile=normalization_quantile,
            clip=True,
            fill_diagonal=False,
            norm_factor=norm_factor,
        )
        S_full_norm = normalize_similarity_matrix(
            S_full_raw,
            normalize=True,
            quantile=normalization_quantile,
            clip=True,
            fill_diagonal=True,
            norm_factor=norm_factor,
        )
    else:
        norm_factor = 1.0
        S_boot_norm = S_boot.copy()
        S_full_norm = S_full_raw.copy()

    # ---------------- ESTADÍSTICO: percentil bootstrap ----------------
    percentile_threshold = vae_params.get("percentile_threshold", 80)
    mu_p = np.percentile(S_boot_norm, percentile_threshold, axis=0)

    # seleccionar bordes confiables
    mask = S_full_norm >= mu_p

    final_adj_norm = np.zeros_like(S_full_norm)
    final_adj_norm[mask] = S_full_norm[mask]
    np.fill_diagonal(final_adj_norm, 1.0)

    final_adj_df = pd.DataFrame(final_adj_norm, index=effectors, columns=effectors)
    S_full_df = pd.DataFrame(S_full_norm, index=effectors, columns=effectors)

    # diagnóstico
    if verbose:
        tri = np.triu_indices(n_effectors, k=1)
        kept = np.sum(final_adj_norm[tri] > 0)
        total_pairs = n_effectors * (n_effectors - 1) // 2
        print(f"[diagnóstico] arcos retenidos: {kept}/{total_pairs} = {kept/total_pairs:.4f}")
        print(f"[normalización] cuantile(0.99) = {norm_factor:.6g}")

    return final_adj_df, S_full_df, S_boot_norm

# Esquema de doble CV
#---------------------------------------------------------------------------
def inner_search_vae(
    matrix,
    param_grid,
    k_folds=5,
    seed=42,
    metric="MAE",
    must_link=None,
    device="cpu",
    verbose=True,
    normalize_matrices=True,
):
    """
    Inner-loop: optimiza hiperparámetros del VAE simple mediante k-fold CV.

    Ahora incluye:
      - MAE_train por fold
      - MAE_val por fold
      - gap = MAE_val_mean - MAE_train_mean (sobreajuste)
    """
    import numpy as np
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(seed)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    results = []

    for params in param_grid:
        if verbose:
            print(f"\n[inner] probando hiperparámetros: {params}")

        fold_metrics_val  = []
        fold_metrics_train = []

        for fold, (train_ix, val_ix) in enumerate(kf.split(matrix.columns)):
            train_matrix = matrix.iloc[:, train_ix]
            val_matrix   = matrix.iloc[:, val_ix]

            # --- ENTRENAR VAE ---
            vae_df, _, _ = bernoulli_vae_network(
                train_matrix,
                must_link=must_link,
                device=device,
                verbose=False,
                normalize_adj=normalize_matrices,
                **params
            )

            # --- EVALUACIÓN EN TRAIN (siempre MAE para detectar sobreajuste) ---
            m_train = evaluate_network_empirical(train_matrix, vae_df)["MAE"]
            fold_metrics_train.append(m_train)

            # --- EVALUACIÓN EN VALIDATION ---
            m_val = evaluate_network_empirical(val_matrix, vae_df)[metric]
            fold_metrics_val.append(m_val)

            if verbose:
                print(
                    f"  fold {fold+1}/{k_folds} → "
                    f"MAE_train: {m_train:.4f} | "
                    f"{metric}_val: {m_val:.4f}"
                )

        # ---- RESUMEN DEL MODELO ----
        mae_train_mean = np.mean(fold_metrics_train)
        mae_val_mean   = np.mean(fold_metrics_val)

        results.append({
            "params": params,
            "metric_value": mae_val_mean,   # alias of mae_val_mean (compatibility)
            "mae_train_mean": mae_train_mean,
            "mae_val_mean": mae_val_mean,
            "gap": mae_val_mean - mae_train_mean,   # indicador de sobreajuste
            "fold_mae_train": fold_metrics_train,
            "fold_metric_val": fold_metrics_val,
        })

        if verbose:
            print(f"  → resumen: MAE_train={mae_train_mean:.4f}, "
                  f"{metric}_val={mae_val_mean:.4f}, gap={mae_val_mean - mae_train_mean:.4f}")

    # seleccionar mejor por métrica de validación
    best = sorted(results, key=lambda x: x["metric_value"])[0]

    if verbose:
        print("\n[inner] mejor configuración:")
        print(best)

    return best, results
# Si mantengo hp ajustados en inner search, se mantiene el MAE?
def evaluate_candidates_stability(matrix, candidates, n_repeats=5, k_folds=5, seed0=42,
                                  normalize_matrices=True):
    import numpy as np
    from sklearn.model_selection import KFold
    rng = np.random.default_rng(seed0)
    summary = []
    for params in candidates:
        mae_vals = []
        mae_trains = []
        for s in range(n_repeats):
            seed = seed0 + s
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            fold_val = []
            fold_train = []
            for tr_idx, val_idx in kf.split(matrix.columns):
                train_mat = matrix.iloc[:, tr_idx]
                val_mat   = matrix.iloc[:, val_idx]
                adj_df, coef_df, extras = bernoulli_vae_network(train_mat, normalize_adj=normalize_matrices,
                                                                verbose=False, device='cpu', seed=seed, **params)
                fold_train.append(evaluate_network_empirical(train_mat, adj_df)['MAE'])
                fold_val.append(evaluate_network_empirical(val_mat, adj_df)['MAE'])
            mae_trains.append(np.mean(fold_train))
            mae_vals.append(np.mean(fold_val))
        summary.append({
            'params': params,
            'mae_val_mean': np.mean(mae_vals),
            'mae_val_std': np.std(mae_vals),
            'mae_train_mean': np.mean(mae_trains),
            'mae_train_std': np.std(mae_trains)
        })
    return summary

def outer_compare_methods(
    matrix,
    best_params,
    must_link=None,
    device="cpu",
    k_folds=5,
    seed=42,
    top_k=20,
    B=100,
    sample_fraction=0.5,
    verbose=True,
    normalize_matrices=True
):
    """
    Outer loop: compara generalización entre VAE simple, bootstrap mean y bootstrap statistical.
    NO reajusta hiperparámetros: usa best_params del inner loop.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    results = []

    for fold, (train_ix, test_ix) in enumerate(kf.split(matrix.columns)):
        train_matrix = matrix.iloc[:, train_ix]
        test_matrix  = matrix.iloc[:, test_ix]

        if verbose:
            print(f"\n[outer] fold {fold+1}/{k_folds}")

        # ============================================================
        # 1) VAE simple
        # ============================================================
        vae_df, _, _ = bernoulli_vae_network(
            train_matrix,
            must_link=must_link,
            normalize_adj=normalize_matrices,
            device=device,
            verbose=False,
            **best_params
        )
        m1 = evaluate_network_empirical(test_matrix, vae_df, top_k=top_k)

        # ============================================================
        # 2) Bootstrap mean
        # ============================================================
        agg_df, _, _, _ = bootstrap_vae_network(
            train_matrix,
            n_bootstrap=B,
            sample_fraction=sample_fraction,
            must_link=must_link,
            vae_params=best_params,
            device=device,
            verbose=False,
            normalize_for_stats=normalize_matrices
        )
        m2 = evaluate_network_empirical(test_matrix, agg_df, top_k=top_k)

        # ============================================================
        # 3) Bootstrap estadístico (mu matrix)
        # ============================================================
        agg_df2, _, S_boot, stats = bootstrap_vae_network(
            train_matrix,
            n_bootstrap=B,
            sample_fraction=sample_fraction,
            must_link=must_link,
            vae_params=best_params,
            device=device,
            verbose=False,
            normalize_for_stats=normalize_matrices
        )
        mu_df = pd.DataFrame(stats["per_edge_mu_matrix"], index=matrix.index, columns=matrix.index)
        m3 = evaluate_network_empirical(test_matrix, mu_df, top_k=top_k)

        results.append({
            "fold": fold+1,
            "vae_simple": m1,
            "bootstrap_mean": m2,
            "bootstrap_stat": m3
        })

    return results

# Comparación y evaluación de co-ocurrencia
#---------------------------------------------------------------------------
def compute_empirical_cooccurrence(matrix):
    """
    Calcula la co-ocurrencia empírica de cada par de filas en matrix (efectores).
    matrix: pd.DataFrame (filas=efectores, cols=cepas), valores 0/1
    """
    mat = matrix.values
    n_eff = mat.shape[0]
    cooc = np.zeros((n_eff, n_eff), dtype=float)
    
    # co-ocurrencia = (# cepas donde ambos están presentes) / (# cepas totales)
    for i in range(n_eff):
        for j in range(n_eff):
            cooc[i, j] = np.sum(mat[i,:] * mat[j,:]) / mat.shape[1]
    
    return pd.DataFrame(cooc, index=matrix.index, columns=matrix.index)

def evaluate_network_empirical(matrix, vae_df, top_k=20):
    """
    Evalúa la calidad de la red inferida por VAE comparando con la co-ocurrencia empírica.
    
    matrix: pd.DataFrame, matriz binaria original (efectores x cepas)
    vae_df: pd.DataFrame, matriz inferida por el VAE (efectores x efectores)
    top_k: int, número de aristas top para solapamiento
    """
    
    # 1) calcular matriz empírica
    emp_df = compute_empirical_cooccurrence(matrix)
    
    # 2) aplanar matrices para correlación y error
    emp_flat = emp_df.values.flatten()
    vae_flat = vae_df.values.flatten()
    
    pear = pearsonr(emp_flat, vae_flat)[0]
    spear = spearmanr(emp_flat, vae_flat)[0]
    mae = np.mean(np.abs(emp_flat - vae_flat))
    rmse = np.sqrt(np.mean((emp_flat - vae_flat)**2))
    
    # 3) top-k aristas (ignorando diagonal)
    def upper_triangle_indices(mat):
        n = mat.shape[0]
        return np.triu_indices(n, k=1)
    
    idx = upper_triangle_indices(emp_df.values)
    emp_upper = emp_df.values[idx]
    vae_upper = vae_df.values[idx]
    
    emp_top_idx = np.argsort(emp_upper)[-top_k:]
    vae_top_idx = np.argsort(vae_upper)[-top_k:]
    
    overlap = len(set(emp_top_idx) & set(vae_top_idx)) / top_k
    
    return {
        "pearson_corr": pear,
        "spearman_corr": spear,
        "MAE": mae,
        "RMSE": rmse,
        f"top_{top_k}_overlap": overlap,
    }

# Representación gráfica de la robustez
#---------------------------------------------------------------------------
def plot_bootstrap_std_histogram(S_boot, figures_output, case, dpi, bins=40, log_scale=False):
    """
    Representa el histograma de la desviación estándar de los arcos
    entre boostraps.
    
    Args:
        S_boot: np.array de shape (B, n_effectors, n_effectors)
        bins: número de bins para el histograma
        log_scale: si True, escala logarítmica en Y
    """
    # calcular std de cada arco
    arc_std = S_boot.std(axis=0)
    # tomar solo i<j para no duplicar arcos
    triu_std = arc_std[np.triu_indices(arc_std.shape[0], k=1)]
    
    fig, ax = plt.subplots(figsize=figsize(W_HALF_XS, W_HALF_XS))
    ax.hist(triu_std, bins=bins, color='#89BBFE', edgecolor='black', linewidth=0.3)
    style(
        ax,
        xlabel='Edge standard deviation (σ)',
        ylabel='Number of edges'
    )
    if log_scale:
        ax.set_yscale('log')
    soft_grid_y(ax)
    # ax.grid(axis='y', linestyle='-', linewidth=0.6, color='#CCCCCC', alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    save(fig, f"{figures_output}/std_{case}", fmt="pdf", dpi=dpi)
    plt.show()

def plot_mean_vs_std_scatter(S_boot, figures_output, case, dpi, k=50, figsize=None):
    """
    Scatter μ–σ de todos los arcos + resaltado de los top-k por magnitud media.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Calcular media y std por arco
    arc_mu = S_boot.mean(axis=0)
    arc_sigma = S_boot.std(axis=0)

    # 2. Tomar solo la parte triangular superior
    iu = np.triu_indices(arc_mu.shape[0], k=1)
    mu_vals = arc_mu[iu]
    sigma_vals = arc_sigma[iu]

    # 3. Identificar los top-k
    idx_sorted = np.argsort(-np.abs(mu_vals))  # ordenar por |μ|
    top_idx = idx_sorted[:k]

    # 4. Crear figura
    panel_size = figsize if figsize is not None else globals()["figsize"](W_HALF_XS, W_HALF_XS)
    fig, ax = plt.subplots(figsize=panel_size)

    # scatter principal
    ax.scatter(mu_vals, sigma_vals, color="lightgray", s=1, alpha=0.7, label="Edges")

    # scatter top-k
    ax.scatter(
        mu_vals[top_idx],
        sigma_vals[top_idx],
        color="#D741A7",
        edgecolor='black',
        linewidth=0.25,
        s=8,
        label=f"Top {k} Edges"
    )

    # Ajustar límites para asegurar visibilidad
    mu_min, mu_max = mu_vals.min(), mu_vals.max()
    sig_min, sig_max = sigma_vals.min(), sigma_vals.max()
    ax.set_xlim(mu_min - 0.02*(mu_max-mu_min), mu_max + 0.02*(mu_max-mu_min))
    ax.set_ylim(sig_min - 0.02*(sig_max-sig_min), sig_max + 0.02*(sig_max-sig_min))

    style(ax, xlabel="Edge mean (μ)", ylabel="Standard deviation (σ)")
    ax.legend(
    loc='upper right',
    frameon=True,
    fontsize=LEGEND,
    facecolor='white',
    edgecolor='none', 
    handlelength=1.2,
    handletextpad=0.6,
    borderpad=0.2,
    labelspacing=0.4)
    save(fig, f"{figures_output}/std_mean_{case}", fmt="pdf", dpi=dpi)
    plt.show()

# Evaluacion de la convergencia de parámetros del bootstrap
#---------------------------------------------------------------------------
def compute_convergence_metrics(S_boot_norm):
    """
    Dado un tensor S_boot_norm de shape (B, n, n),
    devuelve curvas de convergencia para mean y std.
    """
    B, n, _ = S_boot_norm.shape
    mean_corr = np.zeros(B)
    std_corr  = np.zeros(B)
    rmse_mean = np.zeros(B)

    def safe_corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 1.0   # Si no hay variabilidad, la correlación "sucesiva" es perfecta
        return np.corrcoef(x, y)[0,1]


    for k in range(2, B + 1):
        mu_prev = S_boot_norm[:k-1].mean(axis=0).flatten()
        mu_curr = S_boot_norm[:k].mean(axis=0).flatten()
        sigma_prev = S_boot_norm[:k-1].std(axis=0).flatten()
        sigma_curr = S_boot_norm[:k].std(axis=0).flatten()

        mean_corr[k-1] = safe_corr(mu_prev, mu_curr)
        std_corr[k-1]  = safe_corr(sigma_prev, sigma_curr)


        rmse_mean[k-1] = np.sqrt(np.mean((mu_prev - mu_curr)**2))

    return mean_corr, std_corr, rmse_mean

def plot_convergence_metrics(results, figures_output, dpi=400, figsize=(12, 7)):
    """
    Representa las curvas de convergencia del bootstrap para varios valores de
    sample_fraction.
    """
    plot_config = {
        "mean_corr_curve": {
            "ylabel": "Successive mean correlation (Pearson r)",
            "title": "Convergence of the bootstrap mean estimate (μ)",
            "ylim": (0, 1),
            "filename": "convergence_mean_corr",
        },
        "std_corr_curve": {
            "ylabel": "Successive std. dev. correlation (Pearson r)",
            "title": "Convergence of the bootstrap standard deviation estimate (σ)",
            "ylim": (0, 1),
            "filename": "convergence_std_corr",
        },
        "rmse_mean_curve": {
            "ylabel": "Successive mean RMSE",
            "title": "Stability of the bootstrap mean estimate across replicates",
            "ylim": None,
            "filename": "convergence_rmse_mean",
        },
    }

    colors = ['#89BBFE', '#D741A7', '#2B1B88']
    widths = [2,2,2.2]
    styles = ['-', '--', ':']
    for metric_name, cfg in plot_config.items():
        _, ax = plt.subplots(figsize=figsize)

        p = 0
        for r in results:
            y = r[metric_name]
            x = np.arange(1, len(y) + 1)

            label = f"Sample fraction = {r['sample_fraction']}"
            ax.plot(x, y, linewidth=widths[p], linestyle = styles[p] , color = colors[p]  ,label=label)
            p = p + 1

        style(
            ax,
            # title=cfg["title"],
            xlabel="Number of bootstrap replicates",
            ylabel=cfg["ylabel"],
        )
        if cfg["ylim"] is not None:
            ax.set_ylim(*cfg["ylim"])
            
        ax.legend(
            loc='best',
            frameon=True,
            fontsize=12,
            facecolor='white',
            edgecolor='none',
            handlelength=1.2,
            handletextpad=0.6,
            borderpad=0.2,
            labelspacing=0.4
        )
        plt.tight_layout()
        plt.savefig(f"{figures_output}/{cfg['filename']}.png", dpi=dpi, transparent=False)
        plt.savefig(f"{figures_output}/{cfg['filename']}.pdf", bbox_inches="tight", transparent=True)
        plt.show()

# Evaluacion de las aristas pre-establecidas
#---------------------------------------------------------------------------
def evaluar_co_conocidas(matriz, co_conocidas, top_percent=25):
    filas = []

    # generar máscara booleana (upper triangular k=1)
    mask = np.triu(np.ones(matriz.shape, dtype=bool), k=1)
    mask = pd.DataFrame(mask, index=matriz.index, columns=matriz.columns)

    # valores reales upper triangular
    triu_vals = matriz.where(mask).stack()

    n_top = int(len(triu_vals) * top_percent / 100)
    threshold = triu_vals.nlargest(n_top).min()

    for (e1, e2) in co_conocidas:
        weight = matriz.loc[e1, e2]     # valor real
        en_top = weight >= threshold

        filas.append({
            "Efector1": e1,
            "Efector2": e2,
            "Weight": weight,
            "En_top": en_top
        })

    return pd.DataFrame(filas)

def ajustar_co_conocidas_top25(agg_adj_df, tabla_co_conocidas, top_percent=25, eps=1e-6):
    """
    Ajusta los pesos de co_conocidas para asegurar que superen el mismo threshold
    que empleará export_grafo_json_topk.

    Se calcula:
        - solo con valores > 0
        - el k-ésimo mayor valor (igual que export_grafo_json_topk)
    """

    adjusted = agg_adj_df.copy().astype(float)

    # ============================================================
    # 1) Calcular threshold EXACTAMENTE igual que el exportador
    # ============================================================
    weights = adjusted.values.flatten()
    nonzero = weights[weights > 0]

    if len(nonzero) == 0:
        raise ValueError("No hay pesos no nulos en la matriz.")

    k = int(len(nonzero) * (top_percent / 100))
    k = max(k, 1)  # igual que en el exportador

    sorted_nonzero = np.sort(nonzero)
    threshold_export = sorted_nonzero[-k]  # MISMA fórmula que export_grafo_json_topk

    # ============================================================
    # 2) Ajustar co_conocidas para superar ese threshold
    # ============================================================
    for _, row in tabla_co_conocidas.iterrows():
        e1, e2, weight = row['Efector1'], row['Efector2'], row['Weight']

        if e1 in adjusted.index and e2 in adjusted.columns:
            if weight <= threshold_export:
                new_val = threshold_export + eps
                adjusted.loc[e1, e2] = new_val
                adjusted.loc[e2, e1] = new_val  # simetría

    return adjusted

# Post cosas con la matriz de co-ocurrencia
#---------------------------------------------------------------------------
def top5_coocurring_effectors(efector, vae_boot_stat, n=5, plot=False, save_path=None, dpi=300):
    """
    Devuelve los n efectores con mayor co-ocurrencia para un efector dado
    y opcionalmente genera un barplot estilizado.

    Parámetros
    ----------
    efector : str
        Efector de interés.
    vae_boot_stat : pd.DataFrame
        Matriz de co-ocurrencia.
    n : int
        Número de efectores más correlacionados.
    plot : bool
        Si True, muestra el barplot.
    save_path : str o None
        Si se pasa una ruta, guarda la figura en ese archivo.
    dpi : int
        Resolución al guardar la imagen.
    """

    if efector not in vae_boot_stat.index:
        raise ValueError(f"El efector {efector} no existe en vae_boot_stat.")

    # Extraer y ordenar
    cooc = vae_boot_stat.loc[efector].drop(labels=[efector], errors='ignore')
    top = cooc.sort_values(ascending=False).head(n)

    if plot:
        # Preparar datos
        x_labels = top.index
        y_values = top.values

        # Crear barplot con el estilo especificado
        _, ax = plt.subplots(figsize=(16, 6))
        ax.bar(
            x=x_labels,
            height=y_values,
            color="#89BBFE",
            width=0.6
        )
        style(ax, xlabel="Effectors", ylabel="Co-occurrence score")
        plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
        plt.tight_layout()

        # Guardar figura si corresponde
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi)
        plt.show()

    return top


def plot_bootstrap_convergence(results, selected_B, selected_frac,
                               figures_output, dpi=400):
    """
    Plots inner bootstrap convergence curves for multiple combinations
    of B (number of replicates) and sample_fraction.

    Each combination produces a curve showing how the mean pairwise
    correlation of bootstrap matrices evolves as replicates accumulate.
    The selected configuration is highlighted.

    Saves PNG and PDF to figures_output.
    """

    from matplotlib.lines import Line2D

    unique_fracs = sorted(set(r["sample_fraction"] for r in results))
    palette = {f: c for f, c in zip(unique_fracs, ["#89BBFE", "#D741A7", "#43C59E"])}
    lines = {f: c for f, c in zip(unique_fracs, ["-", "--", ":"])}

    legend_elements = [
        Line2D([0], [0], color=palette[f], linewidth=1.5,
               linestyle=lines[f], label=f"frac = {f}")
        for f in unique_fracs
    ]

    metrics = [
        ("mean_corr_curve", "Successive mean Pearson r", "FigureS3a"),
        ("rmse_mean_curve", "Successive mean RMSE",      "FigureS3b"),
    ]

    for metric_key, ylabel_str, fname in metrics:
        fig, ax = plt.subplots(figsize=figsize(W_HALF, H_STD))

        for r in results:
            B = r["B"]
            frac = r["sample_fraction"]
            y = r[metric_key]
            x = np.arange(1, len(y) + 1)

            is_selected = (B == selected_B and frac == selected_frac)
            ax.plot(
                x, y,
                color=palette[frac],
                linewidth=0.75,
                linestyle=lines[frac],
                alpha=1.0,
                label=f"B={B}, frac={frac}" if is_selected else None,
                zorder=3 if is_selected else 1
            )

        soft_grid_y(ax)

        style(
            ax,
            xlabel="Number of bootstrap replicates",
            ylabel=ylabel_str
        )

        if metric_key == "mean_corr_curve":
            y_min = min(min(r["mean_corr_curve"]) for r in results)
            ax.set_ylim(max(0.9, y_min - 0.05), 1.02)

        if metric_key == "mean_corr_curve":
            ax.legend(
                handles=legend_elements,
                fontsize=LEGEND,
                facecolor="white",
                edgecolor="#CCCCCC",
                loc="lower right"
            )
        else:
            ax.legend(
                handles=legend_elements,
                fontsize=LEGEND,
                facecolor="white",
                edgecolor="#CCCCCC",
                loc="upper right"
            )

        fig.tight_layout()
        save(fig, f"{figures_output}/{fname}", fmt="pdf", dpi=dpi)
        save(fig, f"{figures_output}/{fname}", fmt="png", dpi=dpi)
        plt.close(fig)


# =====================================================================
# Held-out ELBO / log-likelihood on the REAL data.
#
# This is a goodness-of-fit measure on the OBSERVED data, complementary to
# seed-to-seed stability (pipeline_main.py) and to recovery accuracy on
# simulated ground truth (simulation_ground_truth.py) -- neither of those
# measures fit to the real, observed effector repertoires.
#
# evaluate_vae_held_out_likelihood() does k-fold cross-validation on the
# real 63 x 150 strain matrix: for each fold, a Bernoulli-VAE is trained
# (via bernoulli_vae_network, UNCHANGED -- same architecture, same training
# loop) on the training strains, and its already-trained encoder/decoder
# (returned in `extras['model']`) is evaluated in eval() mode, WITHOUT any
# further training, on the held-out strains: Bernoulli reconstruction
# log-likelihood and the full ELBO (reconstruction + beta_kl * KL), using the
# encoder's posterior mean (z = mu) for a deterministic, reproducible
# held-out score.
#
# WHY this is complementary to, and distinct from, the ground-truth
# validation: ground-truth recovery (fpr_latent / auc_pr on
# simulated data with a KNOWN precision matrix) measures whether a method
# recovers a known dependency STRUCTURE; held-out ELBO/log-likelihood on the
# REAL data measures goodness of FIT to the observed effector repertoires,
# with no known ground truth to compare against. A configuration can win on
# one criterion and not the other (e.g. a higher-capacity model can fit the
# real data better while overfitting spurious co-occurrences in the
# simulation, or vice versa) -- both are reported, neither is silently
# preferred over the other for the manuscript.
#
# Stratification: leave-one-lineage-out sensitivity (phylogenetic_sensitivity.py)
# already groups the 150 strains into lineage_clusters.csv. For consistency,
# this function stratifies folds by that same lineage label when the file is
# available and every lineage has >= k_folds members; otherwise it falls back
# to a plain random K-fold and says so explicitly (documented via `stratified`
# in the returned rows), rather than silently producing near-empty folds for
# small lineages.
# =====================================================================
def evaluate_vae_held_out_likelihood(matrix, vae_params, k_folds=5, seed=42,
                                     lineage_csv=None, device="cpu", verbose=False):
    """
    K-fold held-out evaluation of a Bernoulli-VAE on the real strain matrix.

    Parameters
    ----------
    matrix : pd.DataFrame (effectors x strains), 0/1
    vae_params : dict
        Passed straight to bernoulli_vae_network (epochs, batch_size, lr,
        latent_dim, hidden_dim, beta_kl, gamma_max, burn_in_frac, ramp_frac).
        `seed` inside vae_params (if present) is ignored in favour of the
        `seed` argument below, so every fold of every config is reproducible
        from a single seed.
    k_folds : int
        Number of folds (default 5).
    seed : int
        Seed for fold assignment AND for bernoulli_vae_network's own
        torch/numpy seeding in each fold (same seed reused across folds is
        fine: each fold trains on a different subset of columns, so the
        resulting models differ even with a fixed seed).
    lineage_csv : str or None
        Path to results/phylogenetic_sensitivity/lineage_clusters.csv. If
        given, found, and every lineage cluster has >= k_folds strains, folds
        are stratified by lineage (StratifiedKFold). Otherwise: plain KFold.
    device : str
    verbose : bool

    Returns
    -------
    pd.DataFrame with one row per fold:
        fold, n_train, n_test, elbo_held_out, log_likelihood_held_out,
        stratified (bool), stratify_note (str)
    """
    import os as _os_local
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import KFold, StratifiedKFold

    strains = matrix.columns.tolist()
    n = len(strains)

    stratified = False
    stratify_note = "random KFold (no lineage_csv given)"
    groups = None
    if lineage_csv is not None and _os_local.path.exists(lineage_csv):
        ldf = pd.read_csv(lineage_csv)
        clu_of = dict(zip(ldf["strain"], ldf["lineage_cluster"]))
        groups = np.array([clu_of.get(s, -1) for s in strains])
        if (groups == -1).any():
            stratify_note = (
                "random KFold (some strains missing from lineage_clusters.csv)"
            )
            groups = None
        else:
            vc = pd.Series(groups).value_counts()
            if vc.min() < k_folds:
                stratify_note = (
                    f"random KFold (smallest lineage cluster has {int(vc.min())} "
                    f"strains < k_folds={k_folds}; stratifying would starve folds)"
                )
                groups = None
            else:
                stratified = True
                stratify_note = "stratified by lineage_clusters.csv"

    if groups is not None:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(np.zeros(n), groups))
    else:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(np.arange(n)))

    vae_params = dict(vae_params)
    vae_params.pop("seed", None)
    beta_kl = float(vae_params.get("beta_kl", 0.1))

    rows = []
    for fold, (train_ix, test_ix) in enumerate(splits):
        train_matrix = matrix.iloc[:, train_ix]
        test_matrix = matrix.iloc[:, test_ix]

        _, _, extras = bernoulli_vae_network(
            train_matrix,
            **vae_params,
            device=device,
            seed=seed,
            verbose=False,
            normalize_adj=False,
        )
        encoder = extras["model"]["encoder"]
        decoder = extras["model"]["decoder"]
        encoder.eval()
        decoder.eval()

        X_test = torch.tensor(test_matrix.values.T.astype(np.float32))
        n_test = X_test.shape[0]
        with torch.no_grad():
            mu, logvar = encoder(X_test)
            z = mu  # posterior mean -> deterministic, reproducible held-out score
            logits = decoder(z)
            recon_sum = F.binary_cross_entropy_with_logits(
                logits, X_test, reduction="sum"
            ).item()
            kl_sum = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            ).item()

        elbo_held_out = -(recon_sum + beta_kl * kl_sum) / n_test
        log_likelihood_held_out = -recon_sum / n_test

        if verbose:
            print(f"  [held-out] fold {fold + 1}/{k_folds}: "
                  f"n_train={len(train_ix)} n_test={n_test} "
                  f"ELBO={elbo_held_out:.4f} logL={log_likelihood_held_out:.4f}")

        rows.append(dict(
            fold=fold,
            n_train=len(train_ix),
            n_test=n_test,
            elbo_held_out=elbo_held_out,
            log_likelihood_held_out=log_likelihood_held_out,
            stratified=stratified,
            stratify_note=stratify_note,
        ))

    return pd.DataFrame(rows)


# =====================================================================
# Ground-truth validation
# ---------------------------------------------------------------------
# Generador de datos binarios sintéticos con estructura de dependencia
# CONOCIDA a nivel de matriz de precisión (grafo gaussiano), pensado para
# comprobar si los métodos de inferencia de red (VAE, Bootstrap Mean, Phi
# crudo, L1 con signo) recuperan BORDES DIRECTOS y NO marcan como borde las
# relaciones puramente INDIRECTAS (cadena A-C-B) ni las inducidas por un
# FACTOR LATENTE compartido H.
#
# NO toca el pipeline principal. Sólo se importa desde simulation_ground_truth.py
# =====================================================================
def generate_synthetic_ground_truth(n_nodes=30, n_samples=150, seed=0,
                                    real_matrix=None,
                                    w_direct=0.72, w_chain=0.45, w_latent=0.36,
                                    structured_prev_band=(0.15, 0.85),
                                    pd_eps=1e-4, max_pd_iter=400,
                                    verbose=False):
    """
    Genera una matriz binaria sintética (n_nodes x n_samples) con estructura de
    dependencia conocida y devuelve el diccionario de verdad conocida.

    Estructura verdadera sobre los n_nodes nodos observables:
      - 3 pares con dependencia DIRECTA positiva  (co-ocurrencia)
      - 3 pares con dependencia DIRECTA negativa  (exclusión)
      - 3 tríos en cadena A-C-B: A-C y C-B directos, A-B SIN borde directo
        (dependencia sólo indirecta a través de C)
      - 2 grupos de 3 nodos {X,Y,Z} que dependen de un factor latente H
        compartido (una H por grupo). H NO aparece en los datos de salida y
        NO hay borde directo entre ningún par dentro de {X,Y,Z}
      - el resto de nodos: sin ninguna conexión (ruido de fondo)
    -> nº de nodos estructurados = 3*2 + 3*2 + 3*3 + 2*3 = 27
       (por eso el valor mínimo utilizable de n_nodes es 28; el defecto es 30
        para dejar algún nodo de ruido. El n_nodes=22 que figuraba en el
        encargo original NO cabe: se documenta la discrepancia.)

    Parámetros
    ----------
    real_matrix : pd.DataFrame (efectores x cepas), 0/1
        Se usa para extraer las prevalencias empíricas (proporción de cepas
        positivas por efector). De esa distribución se muestrean, con
        reemplazo, n_nodes prevalencias objetivo para los nodos sintéticos,
        de modo que el rango de rareza/frecuencia imite al real.
        Los nodos ESTRUCTURADOS se recortan a `structured_prev_band` para que
        la señal no quede destruida por prevalencias extremas; los nodos de
        ruido conservan la prevalencia muestreada sin recortar.
    w_direct, w_chain, w_latent : float
        Magnitud (en valor absoluto) de las entradas NO nulas de la matriz de
        precisión para bordes de par directo, bordes de cadena y brazos
        nodo-H del factor latente, respectivamente. El signo de la entrada de
        precisión se fija dentro de la función (negativo => dependencia
        marginal positiva; positivo => negativa).
    structured_prev_band : (low, high)
        Recorte de prevalencia objetivo para los nodos estructurados.
    pd_eps, max_pd_iter : control del ajuste de definida-positiva.

    Devuelve
    --------
    synthetic_df : pd.DataFrame (n_nodes x n_samples), int 0/1
        index = ['N00', 'N01', ...], columns = ['S000', 'S001', ...]
    ground_truth : dict
        {
          "direct"   : [(i, j, sign), ...]   # pares CON borde directo (con signo +1/-1)
          "indirect" : [(i, j), ...]         # A-B de cada cadena  -> deberían dar negativo
          "latent"   : [(i, j), ...]         # pares dentro de cada {X,Y,Z} -> deberían dar negativo
          "noise"    : [(i, j), ...]         # el resto de pares   -> deberían dar negativo
          "meta"     : {...}                 # prevalencias objetivo, roles, Omega, cadenas, etc.
        }
        Todos los pares son tuplas ordenadas (i < j) con índices 0-based de fila.
    """
    import numpy as np
    import pandas as pd
    from itertools import combinations

    # ---- 0. Configuración de la estructura -------------------------------
    n_dp, n_dn, n_ch, n_lat = 3, 3, 3, 2          # pares+, pares-, cadenas, grupos latentes
    n_struct = n_dp * 2 + n_dn * 2 + n_ch * 3 + n_lat * 3   # = 27
    # -------------------------------------------------------------------
    # POR QUÉ 30 NODOS (y no los 22 del encargo original):
    # the true structure uses DISJOINT nodes:
    #   3 pares directos +  -> 6 nodos
    #   3 pares directos -  -> 6 nodos
    #   3 tríos en cadena   -> 9 nodos
    #   2 grupos latentes   -> 6 nodos (las H son latentes, no cuentan)
    #   total estructurado  -> 27 nodos
    # 22 nodos son físicamente insuficientes para alojar 27 roles sin reutilizar
    # nodos (y reutilizarlos crearía bordes directos no declarados que
    # contaminarían la verdad conocida). Se fija el defecto en 30 = 27 + 3 nodos
    # de ruido; la función rechaza cualquier n_nodes < 28.
    # -------------------------------------------------------------------
    if n_nodes < n_struct + 1:
        raise ValueError(
            f"n_nodes={n_nodes} insuficiente: la estructura pedida necesita "
            f"{n_struct} nodos estructurados + >=1 de ruido (min n_nodes={n_struct + 1}). "
            f"El n_nodes=22 del encargo original no cabe; use n_nodes>=28 (defecto 30)."
        )
    n_noise_nodes = n_nodes - n_struct
    rng = np.random.default_rng(seed)

    # ---- 1. Prevalencias empíricas y muestreo de prevalencias objetivo --
    if real_matrix is None:
        raise ValueError("Se requiere real_matrix (matriz binaria real efectores x cepas).")
    real_vals = np.asarray(real_matrix, dtype=float)
    prev_emp = real_vals.mean(axis=1)                      # una prevalencia por efector real
    prev_emp = prev_emp[(prev_emp > 0) & (prev_emp < 1)]   # descarta efectores triviales
    sampled_prev = rng.choice(prev_emp, size=n_nodes, replace=True)

    # ---- 2. Reparto de roles a los índices de nodo ----------------------
    perm = rng.permutation(n_nodes)
    struct_idx = perm[:n_struct]
    noise_idx = perm[n_struct:]

    p = 0
    dp_pairs = [tuple(struct_idx[p + 2 * k: p + 2 * k + 2]) for k in range(n_dp)]; p += 2 * n_dp
    dn_pairs = [tuple(struct_idx[p + 2 * k: p + 2 * k + 2]) for k in range(n_dn)]; p += 2 * n_dn
    chains = [tuple(struct_idx[p + 3 * k: p + 3 * k + 3]) for k in range(n_ch)]; p += 3 * n_ch      # (A, C, B)
    lat_groups = [tuple(struct_idx[p + 3 * k: p + 3 * k + 3]) for k in range(n_lat)]; p += 3 * n_lat  # (X, Y, Z)

    roles = np.array(["noise"] * n_nodes, dtype=object)
    for a, b in dp_pairs:
        roles[a] = roles[b] = "direct_pos"
    for a, b in dn_pairs:
        roles[a] = roles[b] = "direct_neg"
    for a, c, b in chains:
        roles[a] = roles[b] = "chain_end"
        roles[c] = "chain_mid"
    for x, y, z in lat_groups:
        roles[x] = roles[y] = roles[z] = "latent_leaf"

    # -------------------------------------------------------------------
    # POR QUÉ SE RECORTA LA PREVALENCIA DE LOS NODOS ESTRUCTURADOS:
    # las prevalencias objetivo se muestrean de la distribución empírica real
    # (63 efectores, rango ~0.06-0.99) para que el rango global de rareza imite
    # al real. Pero un nodo estructurado con prevalencia extrema (p.ej. 0.06 o
    # 0.99) se binariza con un umbral gaussiano tan lateral que la dependencia
    # latente inyectada queda destruida por la discretización (la phi resultante
    # colapsa hacia 0), y el par dejaría de poner a prueba a los métodos.
    # Por eso: los nodos de RUIDO conservan la prevalencia muestreada SIN
    # recortar (mantienen el rango real), y solo los nodos ESTRUCTURADOS se
    # recortan a `structured_prev_band` (0.15-0.85 por defecto) para preservar
    # la señal que la validación necesita medir.
    # -------------------------------------------------------------------
    lo, hi = structured_prev_band
    target_prev = sampled_prev.astype(float).copy()
    for idx in struct_idx:
        target_prev[idx] = float(np.clip(target_prev[idx], lo, hi))

    # ---- 3. Matriz de precisión sobre variables latentes (incluye H) ----
    n_H = n_lat
    D = n_nodes + n_H
    H_index = {g: n_nodes + g for g in range(n_lat)}   # índice de la H de cada grupo latente

    Omega = np.eye(D)
    direct_gt, indirect_gt, latent_gt = [], [], []

    def _set(u, v, val):
        Omega[u, v] = val
        Omega[v, u] = val

    # entrada de precisión NEGATIVA  -> covarianza/correlación marginal POSITIVA
    # entrada de precisión POSITIVA  -> correlación marginal NEGATIVA
    for a, b in dp_pairs:
        _set(a, b, -w_direct)
        direct_gt.append((int(min(a, b)), int(max(a, b)), +1))
    for a, b in dn_pairs:
        _set(a, b, +w_direct)
        direct_gt.append((int(min(a, b)), int(max(a, b)), -1))
    for a, c, b in chains:
        _set(a, c, -w_chain)
        _set(c, b, -w_chain)
        direct_gt.append((int(min(a, c)), int(max(a, c)), +1))
        direct_gt.append((int(min(c, b)), int(max(c, b)), +1))
        indirect_gt.append((int(min(a, b)), int(max(a, b))))
    for g, (x, y, z) in enumerate(lat_groups):
        h = H_index[g]
        for node in (x, y, z):
            _set(node, h, -w_latent)
        for u, v in combinations(sorted((int(x), int(y), int(z))), 2):
            latent_gt.append((u, v))

    # ---- definida positiva: si no lo es, encoge las entradas fuera de la diagonal
    off = Omega - np.eye(D)
    scale = 1.0
    for _ in range(max_pd_iter):
        test = np.eye(D) + scale * off
        min_eig = float(np.linalg.eigvalsh(test)[0])
        if min_eig > pd_eps:
            break
        scale *= 0.95
    else:
        raise RuntimeError("No se consiguió una matriz de precisión definida positiva.")
    Omega = np.eye(D) + scale * off
    min_eig_final = float(np.linalg.eigvalsh(Omega)[0])
    is_symmetric = bool(np.allclose(Omega, Omega.T))

    # ---- covarianza -> correlación latente; muestra normal multivariante --
    Sigma = np.linalg.inv(Omega)
    d = np.sqrt(np.diag(Sigma))
    R = Sigma / np.outer(d, d)
    R = 0.5 * (R + R.T)
    Z = rng.multivariate_normal(np.zeros(D), R, size=n_samples)   # (n_samples x D)
    Z_obs = Z[:, :n_nodes]                                        # se DESCARTA H

    # ---- 4. Binarización por umbral gaussiano (cuantil muestral) --------
    B = np.zeros((n_nodes, n_samples), dtype=int)
    for j in range(n_nodes):
        col = Z_obs[:, j]
        thr = np.quantile(col, 1.0 - target_prev[j])   # deja (1-p) por debajo => prevalencia ~ p
        B[j, :] = (col > thr).astype(int)

    # ---- 5. Empaquetado ------------------------------------------------
    all_pairs = set(combinations(range(n_nodes), 2))
    covered = {(i, j) for i, j, _ in direct_gt} | set(indirect_gt) | set(latent_gt)
    noise_gt = sorted(all_pairs - covered)

    node_names = [f"N{ i:02d}" for i in range(n_nodes)]
    sample_names = [f"S{k:03d}" for k in range(n_samples)]
    synthetic_df = pd.DataFrame(B, index=node_names, columns=sample_names)

    ground_truth = {
        "direct": direct_gt,
        "indirect": indirect_gt,
        "latent": latent_gt,
        "noise": noise_gt,
        "meta": {
            "n_nodes": n_nodes,
            "n_samples": n_samples,
            "seed": seed,
            "n_struct_nodes": n_struct,
            "n_noise_nodes": int(n_noise_nodes),
            "n_latent_H": n_H,
            "w_direct": w_direct,
            "w_chain": w_chain,
            "w_latent": w_latent,
            "pd_scale": scale,
            "precision_min_eig": min_eig_final,
            "precision_is_symmetric": is_symmetric,
            "structured_prev_band": tuple(structured_prev_band),
            "target_prevalence": target_prev.tolist(),
            "sampled_prevalence_raw": sampled_prev.tolist(),
            "node_roles": roles.tolist(),
            "direct_pos_pairs": [(int(a), int(b)) for a, b in dp_pairs],
            "direct_neg_pairs": [(int(a), int(b)) for a, b in dn_pairs],
            "chains": [(int(a), int(c), int(b)) for a, c, b in chains],       # (A, C, B)
            "latent_groups": [(int(x), int(y), int(z)) for x, y, z in lat_groups],
            "H_index": H_index,
            "Omega": Omega,          # D x D, incluye las H
            "R_latent": R,           # correlación latente D x D
            "node_names": node_names,
        },
    }

    if verbose:
        emp_prev = B.mean(axis=1)
        print(f"[generate_synthetic_ground_truth] seed={seed} n_nodes={n_nodes} "
              f"pd_scale={scale:.3f} min_eig={min_eig_final:.4f} sym={is_symmetric}")
        print(f"  prevalencia objetivo vs empírica: max|Δ|={np.max(np.abs(emp_prev - target_prev)):.4f}")
        print(f"  #direct={len(direct_gt)} #indirect={len(indirect_gt)} "
              f"#latent={len(latent_gt)} #noise={len(noise_gt)}")

    return synthetic_df, ground_truth


# ─────────────────────────────────────────────────────────────────────────────
# INPUT LOADING  —  shared by pipeline_main.py and build_final_network.py
# ─────────────────────────────────────────────────────────────────────────────
def load_binary_matrix(xlsx_path, sheet_name="Ecoli"):
    """
    Reproduce Block 1 of the pipeline: read the effector-repertoire spreadsheet
    and return the 63 x 150 binary presence/absence matrix
    (rows = effector families, columns = strains).

    Logic is identical to the original ``pipeline_analysis.py`` Block 1:
      * header row 1, effector name in column index 2
      * strain columns from index 9 onward, dropping any whose name matches
        ``suma|total`` (case-insensitive)
      * duplicate effector names disambiguated with ``(a)``, ``(b)`` ... suffixes
      * cells containing ``"pseudogen"`` treated as absent
      * a locus-tag string = present (1), empty = absent (0)
    """
    import numpy as np
    import pandas as pd

    df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=1)
    effector_col = df_raw.columns[2]

    candidate_cols = df_raw.columns[9:]
    excluded_mask = candidate_cols.str.contains("suma|total", case=False, na=False)
    strain_cols = candidate_cols[~excluded_mask]

    df_raw[effector_col] = df_raw[effector_col].astype(str).str.strip()
    df_raw = df_raw[
        df_raw[effector_col].notna()
        & (df_raw[effector_col] != "nan")
        & (df_raw[effector_col] != "")
    ]

    counts = {}
    unique_names = []
    for name in df_raw[effector_col]:
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > 1:
            new_name = f"{name}({chr(96 + counts[name])})"
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

    assert matrix.shape[1] == 150, f"expected 150 strains, got {matrix.shape[1]}"
    assert matrix.sum(axis=0).min() == 4, "minimum repertoire is no longer 4"
    return matrix
