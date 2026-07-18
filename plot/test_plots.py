from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths from the project root.

    This module is expected to be stored in ``plot/test_plots.py``.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[1]
    return project_root / path


def save_test_outputs(
    summary: dict,
    algorithm: str,
    output_dir: str | Path = "plots/test",
) -> dict[str, Path]:
    """Save the raw test summary as JSON."""
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

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_output_dir = (
        _resolve_project_path(output_dir)
        / algorithm.lower()
        / timestamp
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_output_dir / f"{timestamp}_test_results.json"

    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    return {
        "summary": summary_path,
    }
