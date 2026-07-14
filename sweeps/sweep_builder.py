"""Builds one W&B `sweep_config` per (algorithm, task, architecture) triple.

Each sweep targets exactly one architecture so its `parameters` dict stays
flat (see sweeps/config_space.py): the board configuration is a fixed input
to a task, not a search dimension, so board_height/board_width/n_mines are
injected here as fixed `value` entries alongside architecture_name, never as
part of the swept `search_space`.
"""

from __future__ import annotations

from typing import Any

from sweeps.budget import resolve_episode_budget
from sweeps.config_space import build_parameter_space

# Number of reported validation checkpoints Hyperband waits for before it is
# allowed to prune a trial. Kept independent of the episode budget: what
# Hyperband counts is validation *reports* (one per on_validation call), not
# episodes, and how many reports a given episode budget produces depends on
# validation_frequency -- a per-algorithm training setting the sweep doesn't
# necessarily sample. A small fixed floor is a safe default at any board size;
# eta=3 still grows the brackets multiplicatively from there.
DEFAULT_HYPERBAND_MIN_ITER = 3
DEFAULT_HYPERBAND_ETA = 3


def sweep_name(campaign_name: str, task_id: str, algorithm: str, architecture_name: str) -> str:
    return f"{campaign_name}-{algorithm}-{task_id}-{architecture_name}"


def build_sweep_config(
    campaign: dict[str, Any],
    task: dict[str, Any],
    algorithm: str,
    architecture_name: str,
) -> dict[str, Any]:
    """Build a wandb.sweep()-ready config for one (algorithm, task, architecture) triple."""
    swept_parameters = build_parameter_space(algorithm, architecture_name, campaign["search_space"])

    episode_budget = resolve_episode_budget(
        task["board_height"],
        task["board_width"],
        task["n_mines"],
        override=task.get("episode_budget_override"),
    )

    fixed_parameters = {
        "architecture_name": {"value": architecture_name},
        "board_height": {"value": task["board_height"]},
        "board_width": {"value": task["board_width"]},
        "n_mines": {"value": task["n_mines"]},
        "n_episodes": {"value": episode_budget},
    }

    collisions = swept_parameters.keys() & fixed_parameters.keys()
    if collisions:
        raise ValueError(
            f"search_space.{algorithm} must not declare {sorted(collisions)}: "
            "these are fixed per-task/per-architecture values, not search dimensions"
        )

    return {
        "name": sweep_name(campaign["campaign_name"], task["id"], algorithm, architecture_name),
        "method": campaign.get("search_method", "random"),
        "metric": {"name": "search/objective", "goal": "maximize"},
        "early_terminate": {
            "type": "hyperband",
            "min_iter": DEFAULT_HYPERBAND_MIN_ITER,
            "eta": DEFAULT_HYPERBAND_ETA,
        },
        "parameters": {**fixed_parameters, **swept_parameters},
    }
