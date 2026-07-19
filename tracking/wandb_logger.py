"""Logs a completed training run to Weights & Biases.

Reads the metrics already stored in the final checkpoints and
replays them as a single end-of-run upload.
"""

from pathlib import Path

import numpy as np
import torch
import wandb

_COMMON_METRICS = (
    "episode_returns",
    "episode_lengths",
    "episode_wins",
    "training_error",
)

_WIN_RATE_ROLLING_WINDOW = 100


def _win_rate_series(episode_wins: list[int], window: int = _WIN_RATE_ROLLING_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative and trailing-window win rate at each episode index.

    Cumulative smears together policy versions from across the whole run and
    reacts far too slowly on long runs to show current performance; the
    rolling window tracks the recent trend and can also show regressions,
    which a monotonic cumulative average never can.
    """
    wins = np.asarray(episode_wins, dtype=np.float64)
    cumulative_wins = np.cumsum(wins)
    cumulative_win_rate = cumulative_wins / np.arange(1, wins.size + 1)

    window_start = np.maximum(0, np.arange(wins.size) - window + 1)
    padded_cumsum = np.concatenate(([0.0], cumulative_wins))
    window_sum = padded_cumsum[np.arange(1, wins.size + 1)] - padded_cumsum[window_start]
    window_count = np.arange(1, wins.size + 1) - window_start
    rolling_win_rate = window_sum / window_count

    return cumulative_win_rate, rolling_win_rate

_ALGORITHM_METRICS = {
    "dqn": ("epsilon_history", "loss_history"),
    "ppo": ("actor_loss_history", "critic_loss_history", "entropy_history"),
}

# DQN validates every N completed episodes (agents/dqn_agent.py:506-510,
# key "episode"); PPO validates every N rollouts instead
# (agents/ppo_agent.py:673-676, key "rollout") — the two are not on the
# same x-axis, so each algorithm's validation curve gets its own step field.
_VALIDATION_STEP_KEY = {
    "dqn": "episode",
    "ppo": "rollout",
}

# Network weights are read straight from the checkpoint's state_dicts, so
# the parameter count is tracked automatically even when the architecture
# itself is only identified by a hand-set label (or not labeled at all,
# e.g. for backfilled runs).
_STATE_DICT_KEYS = {
    "dqn": {"network_param_count": "online_network_state_dict"},
    "ppo": {
        "actor_param_count": "actor_state_dict",
        "critic_param_count": "critic_state_dict",
    },
}


def _param_count(state_dict: dict) -> int:
    return sum(tensor.numel() for tensor in state_dict.values())


def board_slug(board_height: int, board_width: int, n_mines: int) -> str:
    """Height x width x mine-count, matching this module's existing
    ``test/latest_board_config`` display convention (see ``_log_test_evaluation``)."""
    return f"b{board_height}x{board_width}m{n_mines}"


def run_group(
    algorithm: str,
    architecture_name: str | None,
    board_height: int | None = None,
    board_width: int | None = None,
    n_mines: int | None = None,
) -> str:
    """Group name shared by every seed of one (algorithm, architecture, board)
    variant. Board dims are included so different board sizes (e.g. 6x6 vs
    9x9) don't get overlaid in the same W&B group chart."""
    parts = [algorithm]
    if architecture_name:
        parts.append(architecture_name)
    if board_height is not None and board_width is not None and n_mines is not None:
        parts.append(board_slug(board_height, board_width, n_mines))
    return "-".join(parts)


def variant_artifact_name(
    algorithm: str,
    architecture_name: str | None,
    board_height: int | None = None,
    board_width: int | None = None,
    n_mines: int | None = None,
    seed: int | None = None,
) -> str:
    """One artifact collection per (algorithm, architecture, board, seed)
    variant; best/final checkpoints are logged as versions within it,
    distinguished by alias rather than by separate collection names."""
    parts = [f"model-{algorithm}"]
    if architecture_name:
        parts.append(architecture_name)
    if board_height is not None and board_width is not None and n_mines is not None:
        parts.append(board_slug(board_height, board_width, n_mines))
    if seed is not None:
        parts.append(f"s{seed}")
    return "-".join(parts)


def _test_row(summary: dict) -> dict:
    """Return the aggregate metrics for one held-out evaluation."""
    return {
        "test/board_height": summary["board_height"],
        "test/board_width": summary["board_width"],
        "test/n_mines": summary["num_mines"],
        "test/mine_density": summary["mine_density"],
        "test/n_episodes": summary["n_episodes"],
        "test/wins": summary["wins"],
        "test/win_rate": summary["win_rate"],
        "test/mean_return": summary["mean_return"],
        "test/std_return": summary["std_return"],
        "test/mean_length": summary["mean_length"],
        "test/std_length": summary["std_length"],
    }


def _episode_table(episodes: list[dict]) -> wandb.Table | None:
    """Preserve the full, filterable per-episode result set in W&B."""
    if not episodes:
        return None

    columns = list(episodes[0].keys())
    return wandb.Table(
        columns=columns,
        data=[[episode[column] for column in columns] for episode in episodes],
    )


def _define_test_metrics(test_row: dict) -> None:
    """Give aggregate and per-episode test values independent x-axes."""
    wandb.define_metric("test/eval_index")
    for metric_name in test_row:
        wandb.define_metric(metric_name, step_metric="test/eval_index")

    wandb.define_metric("test/global_episode_index")
    for metric_name in (
        "test/episode_within_eval",
        "test/episode_win",
        "test/episode_return",
        "test/episode_length",
        "test/episode_mine_density",
        "test/episode_cumulative_win_rate",
        "test/episode_cumulative_mean_return",
        "test/episode_cumulative_mean_length",
    ):
        wandb.define_metric(metric_name, step_metric="test/global_episode_index")


def _log_test_evaluation(
    run,
    algorithm: str,
    summary: dict,
    test_row: dict,
    table: wandb.Table | None,
    eval_index: int,
    plot_paths: dict[str, str | Path] | None,
) -> None:
    """Log episode time series, final aggregates, and durable test outputs."""
    _define_test_metrics(test_row)

    episode_offset = int(run.summary.get("test/episode_count", 0))
    cumulative_wins = 0
    cumulative_return = 0.0
    cumulative_length = 0.0
    for episode in summary["episodes"]:
        episode_number = int(episode["episode"])
        cumulative_wins += int(episode["won"])
        cumulative_return += float(episode["return"])
        cumulative_length += float(episode["length"])

        wandb.log({
            "test/global_episode_index": episode_offset + episode_number,
            "test/eval_index": eval_index,
            "test/episode_within_eval": episode_number,
            "test/episode_win": int(episode["won"]),
            "test/episode_return": float(episode["return"]),
            "test/episode_length": int(episode["length"]),
            "test/episode_mine_density": float(episode["mine_density"]),
            "test/episode_cumulative_win_rate": cumulative_wins / episode_number,
            "test/episode_cumulative_mean_return": cumulative_return / episode_number,
            "test/episode_cumulative_mean_length": cumulative_length / episode_number,
        })

    aggregate_row = {**test_row, "test/eval_index": eval_index}
    if table is not None:
        aggregate_row["test/episodes"] = table
        aggregate_row["test/return_distribution"] = wandb.Histogram(
            [episode["return"] for episode in summary["episodes"]]
        )
        aggregate_row["test/length_distribution"] = wandb.Histogram(
            [episode["length"] for episode in summary["episodes"]]
        )
    if plot_paths and "plot" in plot_paths:
        aggregate_row["test/episode_report"] = wandb.Image(str(plot_paths["plot"]))
    wandb.log(aggregate_row)

    if plot_paths and "summary" in plot_paths:
        results_artifact = wandb.Artifact(
            f"{algorithm}-test-evaluation",
            type="evaluation",
        )
        results_artifact.add_file(str(plot_paths["summary"]), name="test_results.json")
        run.log_artifact(results_artifact)

    latest_summary = {
        f"test/latest_{key.removeprefix('test/')}": value
        for key, value in test_row.items()
    }
    run.summary.update({
        "test/eval_count": eval_index + 1,
        "test/episode_count": episode_offset + summary["n_episodes"],
        **latest_summary,
        "test/latest_board_config": (
            f"{summary['board_height']}x{summary['board_width']}"
            f"@{summary['num_mines']}mines"
        ),
    })


def make_live_validation_callback(algorithm: str):
    """builds an on_validation callback that logs win rate to wandb live, during training

    used by main.py for normal (non-sweep) runs: logs under the same "val/win_rate"
    key and step metric that log_run() replays at the end from the checkpoint, so the
    live points and the post-hoc replay end up on the same chart, not two separate ones
    """
    step_key = _VALIDATION_STEP_KEY[algorithm]
    val_step_field = f"val/{step_key}"

    def on_validation(metrics: dict) -> None:
        wandb.define_metric(val_step_field)
        wandb.define_metric("val/win_rate", step_metric=val_step_field)
        wandb.log({
            "val/win_rate": metrics["win_rate"],
            val_step_field: metrics[step_key],
        })

    return on_validation


def log_run(
    algorithm: str,
    checkpoint_path: str | Path,
    best_checkpoint_path: str | Path | None,
    plot_paths: list[str | Path] | None = None,
    project: str = "minesweeper-rl",
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    architecture_name: str | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # The checkpoint filename is already a "%Y-%m-%d-%H-%M-%S" timestamp,
    # so it doubles as a stable, sortable run name
    if name is None:
        name = f"{algorithm}-{checkpoint_path.stem}-train"

    board_config = checkpoint.get("board_config", {})
    seed = checkpoint.get("seed")

    if group is None:
        group = run_group(
            algorithm,
            architecture_name,
            board_config.get("board_height"),
            board_config.get("board_width"),
            board_config.get("n_mines"),
        )

    config = dict(checkpoint["hyperparameters"])
    config.update(board_config)
    if architecture_name is not None:
        config["architecture_name"] = architecture_name

    param_counts = {
        count_key: _param_count(checkpoint[state_dict_key])
        for count_key, state_dict_key in _STATE_DICT_KEYS[algorithm].items()
        if state_dict_key in checkpoint
    }
    config.update(param_counts)
    if len(param_counts) == len(_STATE_DICT_KEYS[algorithm]) and len(param_counts) > 1:
        config["total_param_count"] = sum(param_counts.values())

    # inside a sweep-agent run, wandb.init() was already called before
    # training started (the agent needs it to populate wandb.config with
    # the sampled hyperparameters); re-initializing here would silently
    # start a second, disconnected run instead of reusing it
    owns_run = wandb.run is None
    if owns_run:
        run = wandb.init(
            project=project,
            entity=entity,
            job_type=algorithm,
            group=group,
            name=name,
            tags=tags,
            config=config,
        )
    else:
        run = wandb.run
        run.config.update(config, allow_val_change=True)

    # every checkpoint field is read defensively instead of 
    # assuming it is always present
    wandb.define_metric("episode")
    wandb.define_metric("train/*", step_metric="episode")

    metric_keys = _COMMON_METRICS + _ALGORITHM_METRICS[algorithm]
    n_episodes = len(checkpoint.get("episode_returns", []))
    episode_wins = checkpoint.get("episode_wins", [])
    if episode_wins:
        cumulative_win_rate, rolling_win_rate = _win_rate_series(episode_wins)
    for episode in range(n_episodes):
        row = {
            f"train/{key}": checkpoint[key][episode]
            for key in metric_keys
            if key in checkpoint and episode < len(checkpoint[key])
        }
        if episode < len(episode_wins):
            row["train/cumulative_win_rate"] = float(cumulative_win_rate[episode])
            row[f"train/rolling_win_rate_{_WIN_RATE_ROLLING_WINDOW}"] = float(rolling_win_rate[episode])
        if row:
            row["episode"] = episode + 1
            wandb.log(row)

    # validation lives on its own x-axis instead of piggy-backing on the
    # implicit W&B step, since it is not on the same scale as episodes.
    # skipped when we don't own the run: whoever opened it (main.py,
    # a sweep trial) already streamed each validation point live, on
    # this same "val/win_rate" key, so replaying validation_history here
    # too would just duplicate every point on the chart
    if owns_run:
        validation_history = checkpoint.get("validation_history", [])
        step_key = _VALIDATION_STEP_KEY[algorithm]
        val_step_field = f"val/{step_key}"
        if validation_history:
            wandb.define_metric(val_step_field)
            wandb.define_metric("val/win_rate", step_metric=val_step_field)
            for entry in validation_history:
                wandb.log({
                    "val/win_rate": entry["win_rate"],
                    val_step_field: entry[step_key],
                })

    if "best_validation_win_rate" in checkpoint:
        wandb.summary["best_validation_win_rate"] = checkpoint["best_validation_win_rate"]

    for plot_path in plot_paths or []:
        plot_path = Path(plot_path)
        wandb.log({f"training_curves/{plot_path.stem}": wandb.Image(str(plot_path))})

    # stamp this run's id into the checkpoint file(s) on disk *before*
    # building the artifacts below, so both the artifact copies and the
    # files a later held-out test reads from disk carry it. This is what
    # lets log_test_run() reopen this exact run instead of creating a 
    # disconnected one, without any changes to the agents' save_checkpoint 
    # or the training loop itself
    checkpoint["wandb_run_id"] = run.id
    torch.save(checkpoint, checkpoint_path)

    if best_checkpoint_path is not None and Path(best_checkpoint_path).exists():
        best_checkpoint_path = Path(best_checkpoint_path)
        if best_checkpoint_path == checkpoint_path:
            best_checkpoint = checkpoint
        else:
            best_checkpoint = torch.load(
                best_checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            best_checkpoint["wandb_run_id"] = run.id
            torch.save(best_checkpoint, best_checkpoint_path)

    # final/best are versions of the same per-variant collection,
    # distinguished by alias rather than by separate artifact names
    variant_name = variant_artifact_name(
        algorithm,
        architecture_name,
        board_config.get("board_height"),
        board_config.get("board_width"),
        board_config.get("n_mines"),
        seed,
    )
    artifact_metadata = {
        "algorithm": algorithm,
        "architecture_name": architecture_name,
        "seed": seed,
        "run_id": run.id,
        **board_config,
    }

    final_artifact = wandb.Artifact(
        variant_name, type="model", metadata={**artifact_metadata, "checkpoint_kind": "final"}
    )
    final_artifact.add_file(str(checkpoint_path), name="checkpoint.pt")
    run.log_artifact(final_artifact, aliases=["final"])

    if best_checkpoint_path is not None and Path(best_checkpoint_path).exists():
        best_artifact = wandb.Artifact(
            variant_name, type="model", metadata={**artifact_metadata, "checkpoint_kind": "best"}
        )
        best_artifact.add_file(str(best_checkpoint_path), name="checkpoint.pt")
        run.log_artifact(best_artifact, aliases=["best"])

    # a run opened by a sweep agent is finished by the agent itself once
    # train_entrypoint.py returns; finishing it here would end the run
    # before the agent can move on to the next trial
    if owns_run:
        wandb.finish()


def log_test_run(
    algorithm: str,
    summary: dict,
    checkpoint_path: str | Path,
    project: str = "minesweeper-rl",
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    architecture_name: str | None = None,
    plot_paths: dict[str, str | Path] | None = None,
    force_standalone: bool = False,
) -> None:
    """Logs a held-out test evaluation (see ``tmp/dqn_test.py`` /
    ``tmp/ppo_test.py``) against the *same* W&B run that produced the
    checkpoint being tested, so training curves and every test ever run
    against that checkpoint live in one place with no manual matching.

    This works by reopening the training run via the ``wandb_run_id``
    ``log_run`` stamps into the checkpoint file itself (see there). Each
    call appends its complete per-episode time series plus one indexed
    aggregate row, rather than overwriting a single summary. The same
    checkpoint is expected to be re-tested on purpose (e.g. manual
    generalization checks against board sizes/mine counts different from
    training), and every one of those evaluations should stay visible.

    Checkpoints saved before this field existed (e.g. backfilled
    historical runs) have no ``wandb_run_id`` to resume, so this falls
    back to a standalone run linked only via the W&B Artifacts lineage
    graph (matching "best-checkpoint" artifact content hash).

    ``force_standalone=True`` Used for evaluations that must stay
    fully independent of the training run.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    wandb_run_id = None if force_standalone else checkpoint.get("wandb_run_id")

    test_row = _test_row(summary)
    table = _episode_table(summary["episodes"])

    if wandb_run_id is not None:
        try:
            run = wandb.init(project=project, entity=entity, id=wandb_run_id, resume="must")
        except Exception as error:
            print(
                f"Warning: could not resume wandb run {wandb_run_id!r} "
                f"({error!r}); logging this test as a standalone run instead."
            )
            wandb_run_id = None
        else:
            eval_index = int(run.summary.get("test/eval_count", 0))

            _log_test_evaluation(
                run=run,
                algorithm=algorithm,
                summary=summary,
                test_row=test_row,
                table=table,
                eval_index=eval_index,
                plot_paths=plot_paths,
            )

            run.finish()
            return

    # fallback: no wandb_run_id embedded in this checkpoint, so there is
    # no run to reopen. Same grouping convention as log_run, so this
    # still sits alongside the training runs of the same architecture
    if name is None:
        name = f"{algorithm}-{checkpoint_path.stem}-test"

    seed = checkpoint.get("seed")
    training_board_config = checkpoint.get("board_config", {})

    if group is None:
        group = run_group(
            algorithm,
            architecture_name,
            summary["board_height"],
            summary["board_width"],
            summary["num_mines"],
        )

    config = {
        "board_height": summary["board_height"],
        "board_width": summary["board_width"],
        "n_mines": summary["num_mines"],
        "mine_density": summary["mine_density"],
        "checkpoint_path": summary["checkpoint_path"],
    }
    if architecture_name is not None:
        config["architecture_name"] = architecture_name

    run = wandb.init(
        project=project,
        entity=entity,
        job_type=f"{algorithm}-test",
        group=group,
        name=name,
        tags=[*(tags or []), "test"],
        config=config,
    )

    variant_name = variant_artifact_name(
        algorithm,
        architecture_name,
        training_board_config.get("board_height"),
        training_board_config.get("board_width"),
        training_board_config.get("n_mines"),
        seed,
    )
    checkpoint_artifact = wandb.Artifact(
        variant_name,
        type="model",
        metadata={
            "algorithm": algorithm,
            "architecture_name": architecture_name,
            "seed": seed,
            "checkpoint_kind": "best",
            **training_board_config,
        },
    )
    checkpoint_artifact.add_file(str(checkpoint_path), name="checkpoint.pt")
    run.use_artifact(checkpoint_artifact, aliases=["best"])

    _log_test_evaluation(
        run=run,
        algorithm=algorithm,
        summary=summary,
        test_row=test_row,
        table=table,
        eval_index=0,
        plot_paths=plot_paths,
    )

    run.finish()
