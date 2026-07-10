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
    window: int = 500,
    output_dir: str | Path = "plots",
) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    
    # Generato una sola volta e riutilizzato per cartella e file.
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    run_output_dir = output_dir / timestamp
    run_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = load_training_metrics(
        checkpoint_path
    )

    episode_rewards = metrics[
        "episode_rewards"
    ]

    episode_wins = metrics[
        "episode_wins"
    ]

    episode_lengths = metrics[
        "episode_lengths"
    ]

    # ======================================================
    # Return medio
    # ======================================================

    episodes, rolling_rewards = moving_average(
        episode_rewards,
        window,
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        episodes,
        rolling_rewards,
    )
    plt.xlabel("Episode")
    plt.ylabel("Average return")
    plt.title(
        f"Episode return — moving average "
        f"over {min(window, len(episode_rewards))} episodes"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        run_output_dir / f"{timestamp}_return.png",
        dpi=200,
    )
    plt.close()

    # ======================================================
    # Win rate
    # ======================================================

    episodes, rolling_win_rate = moving_average(
        episode_wins,
        window,
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        episodes,
        rolling_win_rate * 100,
    )
    plt.xlabel("Episode")
    plt.ylabel("Win rate (%)")
    plt.title(
        f"Win rate — moving average "
        f"over {min(window, len(episode_wins))} episodes"
    )
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        run_output_dir / f"{timestamp}_win_rate.png",
        dpi=200,
    )
    plt.close()

    # ======================================================
    # Lunghezza episodi
    # ======================================================

    episodes, rolling_lengths = moving_average(
        episode_lengths,
        window,
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        episodes,
        rolling_lengths,
    )
    plt.xlabel("Episode")
    plt.ylabel("Average episode length")
    plt.title(
        f"Episode length — moving average "
        f"over {min(window, len(episode_lengths))} episodes"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        run_output_dir / f"{timestamp}_length.png",
        dpi=200,
    )
    plt.close()