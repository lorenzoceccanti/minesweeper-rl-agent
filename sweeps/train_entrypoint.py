"""Training function invoked by `wandb agent` for one sweep trial.

`wandb.agent(sweep_id, function=...)` calls its target function with no
arguments per trial, after populating `wandb.config` with this trial's
sampled hyperparameters plus the fixed architecture_name/board_*/n_episodes
values sweep_builder attaches to every trial of a sweep. The algorithm
("dqn"/"ppo") is not itself part of a sweep's parameters (each sweep already
targets exactly one algorithm), so it is bound ahead of time via
`functools.partial(run_trial, algorithm)` when registering with the agent.
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


def load_base_config(path=DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def build_run_config(
    algorithm: str,
    base_config: dict[str, Any],
    sweep_config: dict[str, Any],
) -> dict[str, Any]:
    """Overlay one sweep trial's parameters onto config.yaml's train defaults.

    Same priority order main.py uses for CLI overrides: config.yaml's
    `<algorithm>.train` defaults, then `<algorithm>` root fields
    (architecture_name/hidden_channels/global_features_dim/critic_hidden_size
    /device), then this trial's `sweep_config` (`wandb.config` as a plain
    dict) on top -- the highest-priority layer, since every trial always
    fixes architecture_name/board_*/n_episodes and may also sample training
    hyperparameters that must win over the base config.yaml defaults.
    """
    run_config = dict(base_config[algorithm]["train"])
    inject_algorithm_root_fields(run_config, base_config, algorithm)
    run_config.update(sweep_config)
    return run_config


def make_wandb_callback():
    """Build the concrete on_validation callback, keeping wandb out of agents/*."""

    def on_validation(metrics: dict[str, Any]) -> None:
        wandb.log({"search/objective": metrics["win_rate"], **metrics})

    return on_validation


def run_trial(algorithm: str) -> None:
    run = wandb.init()
    base_config = load_base_config()
    run_config = build_run_config(algorithm, base_config, dict(run.config))
    TRAIN_MODULES[algorithm].run(run_config, on_validation=make_wandb_callback())
