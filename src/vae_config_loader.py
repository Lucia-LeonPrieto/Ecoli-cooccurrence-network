"""
vae_config_loader.py
=====================
Single loader for the VAE hyperparameters stored in src/vae_config.json, so
build_final_network.py, pipeline_main.py, simulation_ground_truth.py and
every other script that fits a VAE can never silently drift apart.

Every such script calls load_vae_params() from here and gets back one dict:
    {**base_params, **selected_params}
i.e. epochs/batch_size/lr/gamma_max/burn_in_frac/ramp_frac (fixed pipeline
constants, unrelated to the hyperparameter search) plus latent_dim/hidden_dim/
beta_kl (the ground-truth-selected winner, see simulation_ground_truth.py).

If any script legitimately needs a DIFFERENT VAE configuration for a specific
sub-analysis, that must be a deliberate, documented exception at the call
site -- never a second hardcoded copy of these three numbers.
"""
import json
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_THIS_DIR, "vae_config.json")


def load_vae_config(config_path=None):
    """Return the raw parsed vae_config.json dict."""
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_vae_params(config_path=None):
    """
    Return a single flat dict combining base_params (fixed pipeline constants)
    and selected_params (the hyperparameter-search winner): exactly what every
    bernoulli_vae_network / bootstrap_vae_network / bootstrap_vae_statistical_network
    call in this repo should receive.
    """
    cfg = load_vae_config(config_path)
    params = dict(cfg["base_params"])
    params.update(cfg["selected_params"])
    return params
