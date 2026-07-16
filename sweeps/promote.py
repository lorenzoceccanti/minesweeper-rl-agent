"""Promotion pipeline: sweep finalists -> multi-seed confirmation retrain -> held-out test.

Adapts tmp/random_search.py's promote/confirm/test stages (select_finalists,
make_confirmation_tasks, run_final_tests) to this project's W&B-native
design: finalists are read from a sweep's finished runs via wandb.Api()
instead of a local screening jsonl, and confirmation/test retrains run as
ordinary W&B runs (tagged stage=confirm) instead of a single-machine
wall-clock scheduling loop.

Two deliberate simplifications versus tmp/random_search.py, both because a
sweep trial's on_validation callback only ever reports `win_rate`
(agents/*.py::evaluate_greedy returns a bare float -- no mean_return is
computed during training-time validation, only during the old script's
separate post-training `evaluate()` call):

  * Screening records (`fetch_finished_screening_records`) and confirmation
    records (`run_confirmation_trial`) both set `validation_mean_return` to
    a fixed 0.0 placeholder. `sweeps.scoring.score`/`select_finalists`/
    `aggregate_confirmation` sort by (win_rate, mean_return) and compute the
    win-rate CI only from win_rate, so this only affects tie-breaking
    between otherwise-equal win rates and the (cosmetic) mean_return column
    in the report -- never the win-rate ranking or the CI itself.
  * A confirmation record's `config_id` is carried over unchanged from its
    source finalist rather than recomputed from the confirm-stage run
    config: n_episodes/agent_seed differ between screening and confirmation
    on purpose (confirm retrains for `confirm_episodes` on a fixed seed),
    and re-hashing those in would make identical hyperparameter/architecture
    configs collide under different config_ids across stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import wandb

from common.paths import resolve_project_path
from evaluation import dqn as evaluate_dqn
from evaluation import ppo as evaluate_ppo
from sweeps.scoring import aggregate_confirmation, config_id, select_finalists
from sweeps.train_entrypoint import build_run_config, load_base_config, make_wandb_callback
from tracking.wandb_logger import run_group
from train import dqn as train_dqn
from train import ppo as train_ppo

TRAIN_MODULES = {"dqn": train_dqn, "ppo": train_ppo}
EVALUATE_MODULES = {"dqn": evaluate_dqn, "ppo": evaluate_ppo}

MISSING_MEAN_RETURN = 0.0

# Terminal states worth ranking. "failed" covers both a Hyperband-pruned
# trial (the common case -- see sweeps/sweep_builder.py's min_iter/eta) and
# a genuine uncaught exception; "crashed" covers a process that died without
# ever calling wandb.finish() (e.g. a background-thread SIGILL). Either way,
# best_win_rate below is what the trial actually achieved before it stopped,
# which is worth ranking regardless of why it stopped. "running"/"queued"
# are excluded: not a stable result yet, could still improve.
_RANKABLE_RUN_STATES = {"finished", "failed", "crashed"}


def promotion_path_for(campaign_name: str) -> Path:
    return resolve_project_path(f"sweeps/registry/{campaign_name}.promotion.json")


def fetch_finished_screening_records(
    sweep_id: str, algorithm: str, api: "wandb.Api | None" = None
) -> list[dict[str, Any]]:
    """Turn one sweep's runs into sweeps.scoring-compatible records.

    Includes Hyperband-pruned and crashed trials, not just naturally
    "finished" ones. `best_win_rate` is logged on every validation report
    (agents/dqn_agent.py, agents/ppo_agent.py -- the running max fed into
    on_validation's metrics dict), so wandb keeps it in run.summary (the
    last logged value) even for a trial killed mid-training. That's unlike
    `best_validation_win_rate`, which tracking/wandb_logger.py only writes
    to wandb.summary from the end-of-run checkpoint -- never reached by a
    pruned trial, hence kept here only as a fallback for runs that predate
    best_win_rate or were never on a sweep controller (e.g. legacy runs).
    """
    api = api or wandb.Api()
    sweep = api.sweep(sweep_id)

    records = []
    for run in sweep.runs:
        if run.state not in _RANKABLE_RUN_STATES:
            continue
        win_rate = run.summary.get("best_win_rate", run.summary.get("best_validation_win_rate"))
        if win_rate is None:
            continue  # never reported a validation checkpoint; nothing to rank
        config = dict(run.config)
        records.append({
            "stage": "screen",
            "algorithm": algorithm,
            "status": "completed",
            "config_id": config_id(config),
            "hyperparameters": config,
            "run_id": run.id,
            "validation_win_rate": float(win_rate),
            "validation_mean_return": MISSING_MEAN_RETURN,
        })
    return records


def select_sweep_finalists(
    sweep_id: str, algorithm: str, count: int, api: "wandb.Api | None" = None
) -> list[dict[str, Any]]:
    records = fetch_finished_screening_records(sweep_id, algorithm, api=api)
    return select_finalists(records, algorithm, count)


def _disjoint_env_seed_start(base_start: int, confirm_episodes: int, seed_index: int) -> int:
    """Give each confirmation seed's own non-overlapping episode/env-seed block.

    Multiplying by `confirm_episodes` guarantees seed_index's block starts
    strictly after every env seed the previous seed_index's confirm run
    could have consumed (one env seed per training episode).
    """
    return base_start + seed_index * confirm_episodes


def _confirm_run_config(
    algorithm: str,
    base_config: dict[str, Any],
    finalist: dict[str, Any],
    confirm_episodes: int,
    seed: int,
    seed_index: int,
) -> dict[str, Any]:
    run_config = build_run_config(algorithm, base_config, finalist["hyperparameters"])
    run_config["agent_seed"] = seed
    run_config["train_env_seed_start"] = _disjoint_env_seed_start(
        run_config["train_env_seed_start"], confirm_episodes, seed_index
    )
    run_config["n_episodes"] = confirm_episodes
    return run_config


def run_confirmation_trial(
    algorithm: str,
    finalist: dict[str, Any],
    seed: int,
    seed_index: int,
    confirm_episodes: int,
    project: str,
    entity: str | None,
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrain one finalist config from scratch on one confirmation seed.

    Runs as a normal, tagged W&B run (stage:confirm) rather than through
    `wandb.agent` -- there is no sweep controlling this trial, so this
    function owns opening and closing the run itself.
    """
    base_config = base_config or load_base_config()
    run_config = _confirm_run_config(algorithm, base_config, finalist, confirm_episodes, seed, seed_index)

    wandb.init(
        project=project,
        entity=entity,
        job_type=f"{algorithm}-confirm",
        group=run_group(
            algorithm,
            run_config["architecture_name"],
            run_config["board_height"],
            run_config["board_width"],
            run_config["n_mines"],
        )
        + "-confirm",
        tags=["sweep", "stage:confirm", f"config:{finalist['config_id']}"],
        config={**run_config, "config_id": finalist["config_id"], "confirm_seed": seed},
    )
    try:
        result = TRAIN_MODULES[algorithm].run(run_config, on_validation=make_wandb_callback())
        win_rate = float(wandb.run.summary.get("best_validation_win_rate", 0.0))
    finally:
        wandb.finish()

    return {
        "stage": "confirm",
        "algorithm": algorithm,
        "status": "completed",
        "config_id": finalist["config_id"],
        "hyperparameters": finalist["hyperparameters"],
        "agent_seed": seed,
        "validation_win_rate": win_rate,
        "validation_mean_return": MISSING_MEAN_RETURN,
        "checkpoint": str(result["best_checkpoint_path"]),
        "board_height": result["board_height"],
        "board_width": result["board_width"],
        "n_mines": result["n_mines"],
    }


def run_confirmation(
    algorithm: str,
    finalists: list[dict[str, Any]],
    confirm_seeds: list[int],
    confirm_episodes: int,
    project: str,
    entity: str | None,
    base_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    base_config = base_config or load_base_config()
    records = []
    for finalist in finalists:
        for seed_index, seed in enumerate(confirm_seeds):
            records.append(
                run_confirmation_trial(
                    algorithm, finalist, seed, seed_index, confirm_episodes, project, entity, base_config
                )
            )
    return records


def select_winner(
    confirm_records: list[dict[str, Any]], algorithm: str, required_seed_count: int
) -> dict[str, Any] | None:
    """Best aggregated confirmation config, excluding any that skipped a seed."""
    aggregates = aggregate_confirmation(confirm_records, algorithm)
    complete = [aggregate for aggregate in aggregates if len(aggregate["records"]) == required_seed_count]
    return complete[0] if complete else None


def run_held_out_test(
    winner: dict[str, Any],
    algorithm: str,
    test_seed_start: int,
    test_episode_count: int,
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Test the winning config's best-performing confirmation replica.

    Reuses evaluation/dqn.py::run / evaluation/ppo.py::run as-is, which
    already logs the result via tracking/wandb_logger.log_test_run against
    the confirm-stage run that produced the checkpoint.
    """
    base_config = base_config or load_base_config()
    best_replica = max(winner["records"], key=lambda record: record["validation_win_rate"])

    test_config = dict(base_config[algorithm]["test"])
    test_config["checkpoint_path"] = best_replica["checkpoint"]
    test_config["board_height"] = best_replica["board_height"]
    test_config["board_width"] = best_replica["board_width"]
    test_config["n_mines"] = best_replica["n_mines"]
    test_config["architecture_name"] = winner["hyperparameters"]["architecture_name"]
    test_config["device"] = base_config.get("main", {}).get("device")
    test_config["test_seed_start"] = test_seed_start
    test_config["test_seed_count"] = test_episode_count

    return EVALUATE_MODULES[algorithm].run(test_config)


class PromotionStore:
    """One JSON file per campaign: sweep_id -> {finalists, confirm_records, winner, test_summary}."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._entries = json.loads(self.path.read_text())

    def set_result(self, sweep_id: str, result: dict[str, Any]) -> None:
        self._entries[sweep_id] = result

    def get(self, sweep_id: str) -> dict[str, Any]:
        return self._entries[sweep_id]

    def results(self) -> dict[str, dict[str, Any]]:
        return dict(self._entries)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2, sort_keys=True, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "PromotionStore":
        return cls(path)


def promote_sweep(
    sweep_id: str,
    algorithm: str,
    task_id: str,
    architecture_name: str,
    promotion: dict[str, Any],
    test: dict[str, Any],
    project: str,
    entity: str | None,
    api: "wandb.Api | None" = None,
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full finalists -> confirm -> (winner) -> held-out test pipeline for one sweep."""
    base_config = base_config or load_base_config()
    finalists = select_sweep_finalists(sweep_id, algorithm, promotion["finalists_per_sweep"], api=api)
    confirm_records = run_confirmation(
        algorithm,
        finalists,
        promotion["confirm_seeds"],
        promotion["confirm_episodes"],
        project,
        entity,
        base_config=base_config,
    )
    winner = select_winner(confirm_records, algorithm, len(promotion["confirm_seeds"]))

    test_summary = None
    if winner is not None:
        test_summary = run_held_out_test(
            winner,
            algorithm,
            test["test_seed_start"],
            test["n_episodes_per_seed"],
            base_config=base_config,
        )

    return {
        "sweep_id": sweep_id,
        "algorithm": algorithm,
        "task_id": task_id,
        "architecture_name": architecture_name,
        "finalists": finalists,
        "confirm_records": confirm_records,
        "winner": winner,
        "test_summary": test_summary,
    }


def _fmt_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def format_report(campaign_name: str, promotion_results: dict[str, dict[str, Any]]) -> str:
    lines = [f"# Promotion report: {campaign_name}", ""]

    for sweep_id, result in sorted(promotion_results.items()):
        lines.append(f"## {sweep_id}")
        lines.append("")
        lines.append(
            f"- algorithm: `{result['algorithm']}`, task: `{result['task_id']}`, "
            f"architecture: `{result['architecture_name']}`"
        )
        lines.append(f"- screening finalists promoted: {len(result['finalists'])}")

        winner = result.get("winner")
        if winner is None:
            lines.append("- **no winner**: no config completed every required confirmation seed")
            lines.append("")
            continue

        lines.append(
            f"- **winner**: config `{winner['config_id']}`, "
            f"confirmation win rate {_fmt_rate(winner['mean_win_rate'])} "
            f"(95% CI {_fmt_rate(winner['ci_low'])} - {_fmt_rate(winner['ci_high'])}, "
            f"n={len(winner['records'])} seeds)"
        )

        test_summary = result.get("test_summary")
        if test_summary is not None:
            lines.append(
                f"- **held-out test**: {test_summary['wins']}/{test_summary['n_episodes']} wins "
                f"({_fmt_rate(test_summary['win_rate'])}), mean return {test_summary['mean_return']:.3f}"
            )
        lines.append("")

    return "\n".join(lines)
