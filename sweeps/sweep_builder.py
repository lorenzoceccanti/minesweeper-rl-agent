"""costruisce il dizionario sweep_config che wandb.sweep() si aspetta

uno per ogni combinazione di algoritmo, task (board) e architettura. ogni
sweep punta a una sola architettura cosi' i parametri restano "piatti"
(vedi sweeps/config_space.py): la board non e' un parametro da cercare ma
un valore fisso del task, quindi board_height/board_width/n_mines vengono
messi qui dentro come "value" bloccato, insieme ad architecture_name, mai
come parte dello spazio di ricerca
"""

from __future__ import annotations

from typing import Any

from sweeps.budget import resolve_episode_budget
from sweeps.config_space import build_parameter_space

# number of reported validation checkpoints Hyperband waits for before it is
# allowed to prune a trial. Kept independent of the episode budget: what
# Hyperband counts is validation *reports* (one per on_validation call), not
# episodes, and how many reports a given episode budget produces depends on
# validation_frequency
DEFAULT_HYPERBAND_MIN_ITER = 3
DEFAULT_HYPERBAND_ETA = 3

# suffix shared by every arch name
_ARCHITECTURE_NAME_SUFFIX = "_3layer_64ch_11in"


# nome leggibile per lo sweep su wandb, tolgo il suffisso comune tanto e' sempre uguale
def sweep_name(campaign_name: str, task_id: str, algorithm: str, architecture_name: str) -> str:
    short_architecture = architecture_name.removesuffix(_ARCHITECTURE_NAME_SUFFIX)
    return f"{campaign_name}-{algorithm}-{short_architecture}-{task_id}"


# main function of the module: assembles the full dict that gets passed straight into wandb.sweep()
def build_sweep_config(
    campaign: dict[str, Any],
    task: dict[str, Any],
    algorithm: str,
    architecture_name: str,
) -> dict[str, Any]:
    """Build a wandb.sweep()-ready config for one (algorithm, task, architecture) triple."""
    # questi sono i parametri che wandb variera' davvero durante la ricerca (learning rate, gamma ecc)
    swept_parameters = build_parameter_space(algorithm, architecture_name, campaign["search_space"])

    # quanti episodi allenare, calcolato in base alla dimensione della board (vedi budget.py)
    episode_budget = resolve_episode_budget(
        task["board_height"],
        task["board_width"],
        task["n_mines"],
        override=task.get("episode_budget_override"),
    )

    # questi invece sono valori bloccati, uguali per ogni trial dello sweep (non vengono cercati)
    fixed_parameters = {
        "architecture_name": {"value": architecture_name},
        "board_height": {"value": task["board_height"]},
        "board_width": {"value": task["board_width"]},
        "n_mines": {"value": task["n_mines"]},
        "n_episodes": {"value": episode_budget},
    }

    # sanity check: nessun parametro fisso deve comparire anche tra quelli cercati, altrimenti e' ambiguo
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
        # hyperband early termination: il controller dello sweep sul cloud di wandb (non il 
        # worker locale) confronta i trial tra loro a ogni checkpoint e killa quelli col 
        # win_rate piu' basso prima che finiscano tutti gli episodi, cosi' il budget temporale
        # si concentra sui config migliori
        "early_terminate": {
            "type": "hyperband",
            "min_iter": DEFAULT_HYPERBAND_MIN_ITER,
            "eta": DEFAULT_HYPERBAND_ETA,
        },
        # unione dei due dict, i parametri fissi prima cosi' in caso di errore e' piu' facile da leggere
        "parameters": {**fixed_parameters, **swept_parameters},
    }
