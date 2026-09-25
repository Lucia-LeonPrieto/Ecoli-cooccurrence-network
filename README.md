# Effector co-occurrence network in *Escherichia coli*

Reproducible pipeline for inferring and validating a co-occurrence network of
type III secretion system (T3SS) effector families across 150 *E. coli*
genomes, using a Bernoulli-VAE together with bootstrap consensus, L1
neighbourhood-selection logistic regression and the Phi coefficient.

This repository is the single source of truth for the manuscript figures and
numbers. Every script reads from `data/` and writes to `results/`; there are no
absolute paths and nothing outside the repository is read.

## Layout

```
.
├── run_all.py                     # run the whole pipeline end to end
├── requirements.txt               # pinned environment (Python 3.12, torch 2.3.1 CPU)
├── LICENSE                        # MIT
├── MANIFEST.csv                   # sha256 + key figure for every file in results/
├── src/
│   ├── pipeline_main.py           # blocks 1-3: intermediates, Bootstrap-Mean net, Table 1
│   ├── build_final_network.py     # the REPORTED network + per-edge confidence
│   ├── inference_functions.py     # all inference + plotting functions (shared library)
│   ├── negative_edges_analysis.py # Table S3: negative-edge interpretability
│   ├── hierarchical_clustering.py # k = 7 modules + dendrogram
│   ├── simulation_ground_truth.py # known-truth validation (7 methods, 35 replicates)
│   ├── phylogenetic_sensitivity.py# leave-one-lineage-out
│   ├── effector_lineage_distribution.py     # per-lineage carrier distribution (Reviewer 1)
│   ├── edge_sign_stability_lineage.py       # VAE-based edge sign stability under LOLO
│   ├── edge_contingency_lineage_check.py    # VAE-free contingency-table stability under LOLO
│   ├── held_out_loglik_beta_kl_comparison.py# Table S4: held-out ELBO / log-likelihood
│   ├── threshold_sensitivity_ground_truth.py# Table S6: accuracy-vs-threshold sweep
│   ├── phylo_corrected_convergence.py       # Mantel-Haenszel phylogeny-corrected clustering
│   ├── cmp_saga_liblinear.py      # saga-vs-liblinear solver agreement (Methods 2.5)
│   ├── extend_tableS2_beta_boundary.py      # Table S2 boundary-probe rows (Methods 2.4)
│   ├── vae_config.json            # single source of truth for VAE hyperparameters
│   ├── vae_config_loader.py       # loads vae_config.json for every script that fits a VAE
│   ├── make_manifest.py           # (re)generate MANIFEST.csv
│   └── figure_style.py            # vendored matplotlib style helper
├── data/
│   ├── input/EffectorRepertoire_Ecoli.xlsx      # INPUT: 63 effector families x 150 strains
│   └── genomes/
│       ├── ecoli_150_distance_matrix.csv        # sourmash (1 - ANI) distance matrix
│       └── ecoli_150_nj_tree.nwk                # neighbour-joining tree
└── results/                       # all outputs (see MANIFEST.csv)
    ├── ground_truth_validation/
    ├── phylogenetic_sensitivity/
    └── phylo_corrected_convergence/
```

## Environment

```
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

`torch==2.3.1` requires `numpy<2` (pinned to 1.26.4). The scripts set
`torch.set_num_threads(1)`: on this small VAE that is both faster (no
oversubscription) and fully deterministic regardless of the host's core count.

## Run

```
python run_all.py
```

Runtime is dominated by the VAE / bootstrap-heavy steps (1, 2, 5, 12); expect
most of a day on CPU. Subsets:

```
python run_all.py --only 3,4      # just negative edges + clustering
python run_all.py --from 3        # resume from step 3
```

Then refresh the checksums:

```
python src/make_manifest.py
```

## Pipeline

| Step | Script | Produces (in `results/`) |
|-----:|--------|--------------------------|
| 1 | `pipeline_main.py` | `01_binary_matrix.csv`, `01_strains_per_effector.csv`, `01_vae_hyperparameter_search_results.csv`, `01_permutation_control_results_rowwise.csv`, `01_bootstrap_convergence_results.csv`, `01_method_stability_comparison_table1_input.csv` (**Table 1**, 7 methods), `01_S_vae_outer.npy`, `01_S_bm_outer.npy`, `01_bootstrap_mean_per_edge_stats.csv`, `01_jaccard_differential_*.csv`, `01_threshold_sensitivity_bootstrap_mean.csv`, `01_network_bootstrap_mean_top30_table1_input.json` |
| 2 | `build_final_network.py` | `02_network_final_signed_similarity.csv` (**reported network**, single-shot VAE, 63×63 signed cosine), `02_edge_confidence.csv` (single-shot value + bootstrap sd over 75 replicas), `02_S_boot_reference_stack.npy` (75,63,63), `02_threshold_sensitivity.csv`, `02_network_final_top30.json` (585 edges, top-30 %) |
| 3 | `negative_edges_analysis.py` | `TableS3.csv` |
| 4 | `hierarchical_clustering.py` | `TableS9.csv`, `04_clustering_structural_balance.csv`, `04_clustering_silhouette_scores.csv`, `04_network_final_clustered.json` (**k = 7**), `figures/FigureS2a.pdf`, `figures/FigureS2b.{pdf,png}` |
| 5 | `simulation_ground_truth.py --phase full` | `ground_truth_validation/05_ground_truth_per_replicate_results.csv`, `05_ground_truth_summary.csv`, `TableS5.csv` (primary criterion: Wilcoxon on `fpr_latent`, VAE vs 4 competitors, raw p + Holm-Bonferroni) |
| 6 | `phylogenetic_sensitivity.py` | `phylogenetic_sensitivity/06_lineage_clusters.csv`, `06_lineage_summary.json`, `06_leave_one_lineage_out.csv`, `06_findings_reviewer_cases.json`, `06_distance_matrix_provenance_copy.csv`, `06_nj_tree_input_provenance_copy.nwk`, `TableS8.csv`, `figures/FigureS1.{pdf,png}` |
| 7 | `effector_lineage_distribution.py` | `07_effector_lineage_distribution.csv`, `07_effector_lineage_distribution_summary.csv` |
| 8 | `edge_sign_stability_lineage.py` | `phylogenetic_sensitivity/08_edge_sign_stability_lineage.csv` |
| 9 | `edge_contingency_lineage_check.py` | `phylogenetic_sensitivity/09_edge_contingency_stability_lineage.csv`, `09_edge_contingency_stability_summary.csv`, `09_edge_contingency_full_dataset_baseline.csv` |
| 10 | `held_out_loglik_beta_kl_comparison.py` | `ground_truth_validation/TableS4.csv`, `TableS4_detail.csv` |
| 11 | `threshold_sensitivity_ground_truth.py` | `ground_truth_validation/TableS6.csv`, `11_threshold_sensitivity_ground_truth_summary.csv` |
| 12 | `phylo_corrected_convergence.py --n-boot-stability 500 --include-l1-diagnostic` | `phylo_corrected_convergence/12_lineage_assignment.csv`, `12_association_MH_logodds_phylo_corrected.csv`, `12_clusters_MH_composition_support.csv`, `12_ari_permutation_test.csv` (**ARI = 0.16, p = 0.0004**, Table S10/S11), `12_l1_stratified_diagnostic_EXCLUDED_*.csv` (Table S12), `figures/12_lineage_dendrogram_qc.{pdf,png}` |
| 13 | `cmp_saga_liblinear.py` | `ground_truth_validation/saga_vs_liblinear_25rep.csv` |
| 14 | `extend_tableS2_beta_boundary.py` | `ground_truth_validation/TableS2.csv` (adds the 2 boundary-probe rows) |

`results/figures/Graph_Clustered.png` is not produced by any script: it is a
manual Cytoscape layout of `04_network_final_clustered.json` (opened
directly in Cytoscape, exported as PNG), kept for the main-text network
figure. `results/TableS7.csv` is likewise not code-derived -- a
hand-maintained table of method descriptions. Neither is regenerated by
`run_all.py`.

### The reported network vs the Bootstrap-Mean network

Two networks are built and both are kept:

* **`02_network_final*` — the network reported in the manuscript.** A single
  Bernoulli-VAE fitted on the full 150-strain matrix (latent 32, hidden 64,
  `beta_kl` 0.01, 300 epochs — `src/vae_config.json`); signed raw cosine similarity between decoder
  weight rows. Per-edge confidence (`02_edge_confidence.csv`) is the standard
  deviation across 75 bootstrap resamples of the strains — the bootstrap stack
  is **not** averaged into the network.
* **`01_network_bootstrap_mean_top30_table1_input.json` — the Bootstrap-Mean network.** The mean
  of the same 75 replicas. It is one of the seven methods compared in Table 1
  (`01_method_stability_comparison_table1_input.csv`); it is not the reported network.

## Input data

`data/input/EffectorRepertoire_Ecoli.xlsx`, sheet **`Ecoli`**, read with
`header=1`: effector name in column index 2, one binary presence column per
strain from index 9 onward (locus tag = present, empty = absent, `pseudogen…` =
absent). Strain columns matching `suma|total` are dropped by name (this file has
none). The binary matrix is 63 effector families × 150 strains; the smallest
repertoire is `LT903847` (4 effectors).
