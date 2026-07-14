"""Shared config-merge logic between main.py and sweeps/train_entrypoint.py.

Extracted from main.py, where architecture_name/hidden_channels/
global_features_dim/critic_hidden_size live once at config[algorithm] root
(not duplicated under train/test) and device lives at config.main -- both
paths that build a run_config (the CLI and the future wandb-agent entrypoint)
must inject these the same way, or a sweep run could silently pick up a
different architecture/device than a manual run from the same config.yaml.
"""

from __future__ import annotations

from typing import Any

ALGORITHM_ROOT_FIELDS = ("hidden_channels", "global_features_dim", "critic_hidden_size")


def inject_algorithm_root_fields(
    run_config: dict[str, Any],
    config: dict[str, Any],
    algorithm: str,
) -> dict[str, Any]:
    """Overlay architecture/device fields from `config` root onto `run_config`.

    Mutates and returns `run_config`. `architecture_name` is always required;
    `hidden_channels`/`global_features_dim`/`critic_hidden_size` are copied
    only if present under `config[algorithm]` (not every algorithm/network
    combination uses all three -- e.g. critic_hidden_size is PPO-only).
    """
    alg_root = config[algorithm]
    run_config["architecture_name"] = alg_root["architecture_name"]

    for key in ALGORITHM_ROOT_FIELDS:
        if key in alg_root:
            run_config[key] = alg_root[key]

    run_config["device"] = config.get("main", {}).get("device")
    return run_config
