from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime


def moving_average(
    values: list[float],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    values_array = np.asarray(
        values,
        dtype=float,
    )

    if values_array.size == 0:
        raise ValueError(
            "Cannot plot an empty metric."
        )

    actual_window = min(
        window,
        len(values_array),
    )

    kernel = (
        np.ones(actual_window)
        / actual_window
    )

    averaged_values = np.convolve(
        values_array,
        kernel,
        mode="valid",
    )

    episodes = np.arange(
        actual_window,
        len(values_array) + 1,
    )

    return episodes, averaged_values


def load_training_metrics(
    checkpoint_path: str | Path,
) -> dict:
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if "algorithm" not in checkpoint:
        raise KeyError(
            "Missing 'algorithm' field in checkpoint."
        )

    algorithm = checkpoint["algorithm"]

    common_metrics = (
        "episode_rewards",
        "episode_lengths",
        "episode_wins",
        "training_error",
    )

    if algorithm == "dqn":
        algorithm_metrics = (
            "epsilon_history",
            "loss_history",
        )

    elif algorithm == "ppo":
        algorithm_metrics = (
            "actor_loss_history",
            "critic_loss_history",
        )

    else:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}"
        )

    required_metrics = (
        common_metrics
        + algorithm_metrics
    )

    missing_metrics = [
        metric
        for metric in required_metrics
        if metric not in checkpoint
    ]

    if missing_metrics:
        raise KeyError(
            f"Missing metrics in {algorithm.upper()} checkpoint: "
            + ", ".join(missing_metrics)
        )

    metrics = {
        "algorithm": algorithm,
    }

    for metric in required_metrics:
        metrics[metric] = checkpoint[metric]

    return metrics


def plot_training_from_checkpoint(
    checkpoint_path: str | Path,
    board_height: int,
    board_width: int,
    num_mines: int,
    window: int = 500,
    output_dir: str | Path = "plots",
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    if board_height <= 0 or board_width <= 0:
        raise ValueError(
            "Board dimensions must be greater than zero."
        )

    if num_mines < 0:
        raise ValueError(
            "The number of mines cannot be negative."
        )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d-%H-%M-%S"
    )

    run_output_dir = output_dir / timestamp
    run_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = load_training_metrics(
        checkpoint_path
    )

    algorithm = metrics[
        "algorithm"
    ].upper()

    episode_rewards = metrics[
        "episode_rewards"
    ]

    episode_wins = metrics[
        "episode_wins"
    ]

    episode_lengths = metrics[
        "episode_lengths"
    ]

    reward_episodes, rolling_rewards = moving_average(
        episode_rewards,
        window,
    )

    win_episodes, rolling_win_rate = moving_average(
        episode_wins,
        window,
    )

    length_episodes, rolling_lengths = moving_average(
        episode_lengths,
        window,
    )

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 18),
    )

    header_text = (
        f"[{algorithm}] - "
        f"board: {board_height}x{board_width} - "
        f"mines: {num_mines}"
    )

    figure.text(
        0.5,
        0.975,
        header_text,
        ha="center",
        va="top",
        fontsize=13,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "whitesmoke",
            "edgecolor": "gray",
        },
    )

    # ======================================================
    # Return medio
    # ======================================================

    axes[0].plot(
        reward_episodes,
        rolling_rewards,
    )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average return")
    axes[0].set_title(
        f"Episode return — moving average "
        f"over {min(window, len(episode_rewards))} episodes"
    )
    axes[0].grid(
        True,
        alpha=0.3,
    )

    # ======================================================
    # Win rate
    # ======================================================

    axes[1].plot(
        win_episodes,
        rolling_win_rate * 100,
    )
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Win rate (%)")
    axes[1].set_title(
        f"Win rate — moving average "
        f"over {min(window, len(episode_wins))} episodes"
    )
    axes[1].set_ylim(
        0,
        100,
    )
    axes[1].grid(
        True,
        alpha=0.3,
    )

    # ======================================================
    # Lunghezza episodi
    # ======================================================

    axes[2].plot(
        length_episodes,
        rolling_lengths,
    )
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Average episode length")
    axes[2].set_title(
        f"Episode length — moving average "
        f"over {min(window, len(episode_lengths))} episodes"
    )
    axes[2].grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.95),
        h_pad=3.0,
    )

    output_path = (
        run_output_dir
        / f"{timestamp}_training_metrics.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
    )
    plt.close(
        figure
    )

    return output_path
