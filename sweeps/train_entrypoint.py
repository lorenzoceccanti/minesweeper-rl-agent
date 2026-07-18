"""funzione che wandb chiama per far partire un singolo trial dello sweep

wandb.agent(sweep_id, function=...) chiama questa funzione senza argomenti,
ma prima riempie wandb.config con gli iperparametri campionati per quel
trial piu' i valori fissi (architecture_name/board_*/n_episodes) che
sweep_builder attacca a ogni trial.
"""

from __future__ import annotations

from typing import Any

import wandb
import yaml

from common.config_merge import inject_algorithm_root_fields
from common.paths import resolve_project_path
from train import dqn as train_dqn
from train import ppo as train_ppo

TRAIN_MODULES = {"dqn": train_dqn, "ppo": train_ppo}
DEFAULT_CONFIG_PATH = resolve_project_path("config.yaml")


# carica il config.yaml del progetto (quello con tutti i default di training)
def load_base_config(path=DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


# overlays the sweep trial's hyperparameters on top of the yaml defaults, highest priority wins
def build_run_config(
    algorithm: str,
    base_config: dict[str, Any],
    sweep_config: dict[str, Any],
) -> dict[str, Any]:
    
    run_config = dict(base_config[algorithm]["train"])  # parti dai default dell'algoritmo
    inject_algorithm_root_fields(run_config, base_config, algorithm)  # aggiungi i campi di architettura
    run_config.update(sweep_config)  # e infine sovrascrivi con quello che ha campionato wandb
    return run_config


# builds the actual callback function for agents/*.py 
def make_wandb_callback():

    def on_validation(metrics: dict[str, Any]) -> None:
        # "search/objective" e' la metrica che wandb usa per decidere quanto e' buono il trial
        wandb.log({"search/objective": metrics["win_rate"], **metrics})

    return on_validation


# funzione vera e propria che wandb.agent chiama per ogni singolo trial dello sweep
def run_trial(algorithm: str) -> None:
    # tagged so sweep trials can be easily filtered in web UI
    run = wandb.init(tags=["sweep"])
    base_config = load_base_config()
    run_config = build_run_config(algorithm, base_config, dict(run.config))
    TRAIN_MODULES[algorithm].run(run_config, on_validation=make_wandb_callback())
