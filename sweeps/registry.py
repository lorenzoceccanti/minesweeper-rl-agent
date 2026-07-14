"""Tracks W&B sweep IDs against the code state each sweep was registered under.

A worker can start hours or days after `register`, on a different machine.
If the code has drifted since registration (a different commit, uncommitted
changes, or an edited agents/models/train file), trials it runs are not
comparable to trials from before the drift -- silently mixing them would
corrupt scoring/promotion. The registry is the source of truth `worker`
checks against before running any sweep.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.paths import resolve_project_path

TRACKED_SOURCE_FILES = (
    "agents/dqn_agent.py",
    "agents/ppo_agent.py",
    "models/factory.py",
    "train/dqn.py",
    "train/ppo.py",
)


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=resolve_project_path("."),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def current_git_commit() -> str:
    return _run_git("rev-parse", "HEAD")


def current_git_dirty() -> bool:
    return bool(_run_git("status", "--porcelain"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_source_hashes(tracked_files: tuple[str, ...] = TRACKED_SOURCE_FILES) -> dict[str, str]:
    return {
        relative_path: _file_sha256(resolve_project_path(relative_path))
        for relative_path in tracked_files
    }


def build_code_state() -> dict[str, Any]:
    return {
        "git_commit": current_git_commit(),
        "git_dirty": current_git_dirty(),
        "source_hashes": current_source_hashes(),
    }


def code_state_mismatches(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Human-readable list of what drifted; empty if the code states match."""
    mismatches = []
    if recorded.get("git_commit") != current.get("git_commit"):
        mismatches.append(
            f"git_commit: registered={recorded.get('git_commit')!r} current={current.get('git_commit')!r}"
        )
    if recorded.get("git_dirty") != current.get("git_dirty"):
        mismatches.append(
            f"git_dirty: registered={recorded.get('git_dirty')!r} current={current.get('git_dirty')!r}"
        )

    recorded_hashes = recorded.get("source_hashes", {})
    current_hashes = current.get("source_hashes", {})
    for relative_path in sorted(set(recorded_hashes) | set(current_hashes)):
        if recorded_hashes.get(relative_path) != current_hashes.get(relative_path):
            mismatches.append(f"source file changed: {relative_path}")

    return mismatches


class Registry:
    """One JSON file per campaign: sweep_id -> {code state, task, algorithm, architecture}."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._entries = json.loads(self.path.read_text())

    def register(
        self,
        sweep_id: str,
        *,
        task_id: str,
        algorithm: str,
        architecture_name: str,
        code_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "task_id": task_id,
            "algorithm": algorithm,
            "architecture": architecture_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(code_state if code_state is not None else build_code_state()),
        }
        self._entries[sweep_id] = entry
        return entry

    def get(self, sweep_id: str) -> dict[str, Any]:
        return self._entries[sweep_id]

    def sweep_ids(self) -> list[str]:
        return list(self._entries)

    def entries_for_algorithms(self, algorithms: list[str]) -> dict[str, dict[str, Any]]:
        return {
            sweep_id: entry
            for sweep_id, entry in self._entries.items()
            if entry["algorithm"] in algorithms
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        return cls(path)
