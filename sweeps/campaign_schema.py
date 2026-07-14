"""Validates a campaign YAML's structure before it is expanded into sweeps.

Mistakes here (a typo'd architecture name, a duplicate task id, a search_space
key that collides with a fixed task parameter) should fail at `cli.py
validate` time with a readable message, not surface as a cryptic wandb.sweep()
error or -- worse -- silently waste trials once workers are already running.
"""

from __future__ import annotations

from typing import Any

from sweeps.config_space import KNOWN_ARCHITECTURES, build_parameter_space

KNOWN_ALGORITHMS = frozenset({"dqn", "ppo"})
KNOWN_SEARCH_METHODS = frozenset({"random", "grid", "bayes"})

REQUIRED_TOP_LEVEL_KEYS = (
    "campaign_name",
    "base_config",
    "tasks",
    "architectures",
    "search_space",
    "worker_profiles",
    "promotion",
    "test",
)

REQUIRED_TASK_KEYS = ("id", "board_height", "board_width", "n_mines")


def validate_campaign(campaign: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors; empty means valid."""
    errors: list[str] = []

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in campaign:
            errors.append(f"missing required top-level key: {key!r}")

    if errors:
        # everything below assumes these keys exist; fail fast
        return errors

    errors.extend(_validate_tasks(campaign["tasks"]))
    errors.extend(_validate_architectures(campaign["architectures"]))

    search_method = campaign.get("search_method", "random")
    if search_method not in KNOWN_SEARCH_METHODS:
        errors.append(
            f"search_method {search_method!r} must be one of {sorted(KNOWN_SEARCH_METHODS)}"
        )

    errors.extend(_validate_search_space(campaign["architectures"], campaign["search_space"]))
    errors.extend(_validate_worker_profiles(campaign["worker_profiles"], campaign["architectures"]))
    errors.extend(_validate_promotion(campaign["promotion"]))
    errors.extend(_validate_test(campaign["test"]))

    return errors


def _validate_tasks(tasks: Any) -> list[str]:
    errors = []
    if not isinstance(tasks, list) or not tasks:
        return ["'tasks' must be a non-empty list"]

    seen_ids = set()
    for task in tasks:
        missing = [key for key in REQUIRED_TASK_KEYS if key not in task]
        if missing:
            errors.append(f"task {task!r} missing required keys: {missing}")
            continue
        if task["id"] in seen_ids:
            errors.append(f"duplicate task id: {task['id']!r}")
        seen_ids.add(task["id"])
        if task["n_mines"] >= task["board_height"] * task["board_width"]:
            errors.append(f"task {task['id']!r}: n_mines must be less than board_height*board_width")

    return errors


def _validate_architectures(architectures: Any) -> list[str]:
    errors = []
    if not isinstance(architectures, dict) or not architectures:
        return ["'architectures' must be a non-empty mapping of algorithm -> list of architecture names"]

    for algorithm, archs in architectures.items():
        if algorithm not in KNOWN_ALGORITHMS:
            errors.append(f"unknown algorithm {algorithm!r}; known algorithms: {sorted(KNOWN_ALGORITHMS)}")
        if not archs:
            errors.append(f"architectures.{algorithm} must list at least one architecture")
        for architecture_name in archs or []:
            if architecture_name not in KNOWN_ARCHITECTURES:
                errors.append(
                    f"architectures.{algorithm}: unknown architecture {architecture_name!r}; "
                    f"known architectures: {sorted(KNOWN_ARCHITECTURES)}"
                )

    return errors


def _validate_search_space(architectures: dict[str, Any], search_space: dict[str, Any]) -> list[str]:
    errors = []
    for algorithm, archs in architectures.items():
        for architecture_name in archs or []:
            if architecture_name not in KNOWN_ARCHITECTURES:
                continue  # already reported by _validate_architectures
            try:
                build_parameter_space(algorithm, architecture_name, search_space)
            except ValueError as error:
                errors.append(str(error))

    return errors


def _validate_worker_profiles(worker_profiles: Any, architectures: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(worker_profiles, dict) or not worker_profiles:
        return ["'worker_profiles' must be a non-empty mapping of profile name -> settings"]

    for profile_name, profile in worker_profiles.items():
        if "max_concurrent_runs" not in profile or profile["max_concurrent_runs"] <= 0:
            errors.append(f"worker_profiles.{profile_name}.max_concurrent_runs must be a positive integer")
        for algorithm in profile.get("algorithms", []):
            if algorithm not in architectures:
                errors.append(
                    f"worker_profiles.{profile_name} references algorithm {algorithm!r} "
                    "not present in 'architectures'"
                )

    return errors


def _validate_promotion(promotion: Any) -> list[str]:
    errors = []
    if "finalists_per_sweep" not in promotion or promotion["finalists_per_sweep"] <= 0:
        errors.append("promotion.finalists_per_sweep must be a positive integer")
    if "confirm_episodes" not in promotion or promotion["confirm_episodes"] <= 0:
        errors.append("promotion.confirm_episodes must be a positive integer")
    if not promotion.get("confirm_seeds"):
        errors.append("promotion.confirm_seeds must be a non-empty list")

    return errors


def _validate_test(test: Any) -> list[str]:
    errors = []
    if "n_episodes_per_seed" not in test or test["n_episodes_per_seed"] <= 0:
        errors.append("test.n_episodes_per_seed must be a positive integer")
    if "test_seed_start" not in test:
        errors.append("test.test_seed_start is required")

    return errors
