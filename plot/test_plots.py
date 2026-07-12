from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths from the project root.

    This module is expected to be stored in ``plot/test_plots.py``.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[1]
    return project_root / path


def _moving_average(
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x coordinates and moving-average values."""
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")

    if values.size == 0:
        raise ValueError("Cannot compute a moving average on empty data.")

    if window <= 0:
        raise ValueError("window must be greater than zero.")

    actual_window = min(window, values.size)
    kernel = np.ones(actual_window, dtype=np.float64) / actual_window

    averaged_values = np.convolve(
        values,
        kernel,
        mode="valid",
    )

    x_values = np.arange(
        actual_window,
        values.size + 1,
    )

    return x_values, averaged_values


def save_test_outputs(
    summary: dict,
    algorithm: str,
    board_height: int,
    board_width: int,
    num_mines: int,
    window: int = 50,
    output_dir: str | Path = "plots/test",
) -> dict[str, Path]:
    """Save a PNG report and the raw test summary as JSON.

    The PNG contains three vertically stacked plots:
    episode return, episode length and rolling win rate.
    """
    episode_results = summary.get("episodes", [])

    if not episode_results:
        raise ValueError("The test summary contains no episode results.")

    required_summary_keys = {
        "n_episodes",
        "wins",
        "win_rate",
        "mean_return",
        "std_return",
        "mean_length",
        "std_length",
    }

    missing_keys = required_summary_keys.difference(summary)
    if missing_keys:
        raise KeyError(
            "Missing keys in test summary: "
            + ", ".join(sorted(missing_keys))
        )

    returns = np.asarray(
        [result["return"] for result in episode_results],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [result["length"] for result in episode_results],
        dtype=np.float64,
    )
    wins = np.asarray(
        [result["won"] for result in episode_results],
        dtype=np.float64,
    )

    episode_indices = np.arange(1, len(episode_results) + 1)
    actual_window = min(window, len(episode_results))

    average_x, average_returns = _moving_average(
        returns,
        actual_window,
    )
    _, average_lengths = _moving_average(
        lengths,
        actual_window,
    )
    _, rolling_win_rate = _moving_average(
        wins,
        actual_window,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_output_dir = (
        _resolve_project_path(output_dir)
        / algorithm.lower()
        / timestamp
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = run_output_dir / f"{timestamp}_test.png"
    summary_path = run_output_dir / f"{timestamp}_test_results.json"

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 14),
        sharex=True,
    )

    axes[0].plot(
        episode_indices,
        returns,
        alpha=0.25,
        label="Episode return",
    )
    axes[0].plot(
        average_x,
        average_returns,
        label=f"Moving average ({actual_window})",
    )
    axes[0].axhline(
        summary["mean_return"],
        linestyle="--",
        label="Mean return",
    )
    axes[0].set_title("Test episode return")
    axes[0].set_ylabel("Return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        episode_indices,
        lengths,
        alpha=0.25,
        label="Episode length",
    )
    axes[1].plot(
        average_x,
        average_lengths,
        label=f"Moving average ({actual_window})",
    )
    axes[1].axhline(
        summary["mean_length"],
        linestyle="--",
        label="Mean length",
    )
    axes[1].set_title("Test episode length")
    axes[1].set_ylabel("Steps")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        average_x,
        rolling_win_rate * 100.0,
        label=f"Rolling win rate ({actual_window})",
    )
    axes[2].axhline(
        summary["win_rate"] * 100.0,
        linestyle="--",
        label=f"Overall win rate: {summary['win_rate']:.2%}",
    )
    axes[2].set_title("Test win rate")
    axes[2].set_xlabel("Test episode")
    axes[2].set_ylabel("Win rate (%)")
    axes[2].set_ylim(0.0, 100.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.suptitle(
        f"[{algorithm.upper()}] - TEST - "
        f"board: {board_height}x{board_width} - "
        f"mines: {num_mines}",
        fontsize=14,
    )

    summary_text = (
        f"episodes: {summary['n_episodes']}   |   "
        f"wins: {summary['wins']}   |   "
        f"win rate: {summary['win_rate']:.2%}   |   "
        f"mean return: {summary['mean_return']:.3f} ± "
        f"{summary['std_return']:.3f}   |   "
        f"mean length: {summary['mean_length']:.3f} ± "
        f"{summary['std_length']:.3f}"
    )

    figure.text(
        0.5,
        0.955,
        summary_text,
        horizontalalignment="center",
        verticalalignment="top",
    )

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    figure.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    return {
        "plot": plot_path,
        "summary": summary_path,
    }
