"""tiene traccia degli sweep id di wandb insieme allo stato del codice con cui sono stati registrati

un worker puo' partire ore o giorni dopo il 'register', magari su un'altra
macchina. se nel frattempo il codice e' cambiato (commit diverso, modifiche
non committate, un file di agents/models/train editato), i trial di quel
worker non sono piu' confrontabili con quelli di prima -- mischiarli senza
accorgersene rovinerebbe scoring e promotion. il registry e' la fonte di
verita' che 'worker' controlla prima di far girare qualsiasi sweep
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.paths import resolve_project_path

# file "sensibili": se cambiano dopo che uno sweep e' stato registrato, i trial diventano
# non piu' confrontabili con quelli fatti prima (logica di training/agenti diversa)
TRACKED_SOURCE_FILES = (
    "agents/dqn_agent.py",
    "agents/ppo_agent.py",
    "models/factory.py",
    "train/dqn.py",
    "train/ppo.py",
)


# generic helper to run a git command and get back the cleaned up stdout
def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=resolve_project_path("."),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# hash del commit attuale su cui si trova il repo
def current_git_commit() -> str:
    return _run_git("rev-parse", "HEAD")


# true se ci sono modifiche non committate (working tree sporco)
def current_git_dirty() -> bool:
    return bool(_run_git("status", "--porcelain"))


# hashes a single file's content, catches edits even when there's no new commit
def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# calcola l'hash di ogni file "tracciato" cosi' si puo' confrontare dopo se qualcosa e' cambiato
def current_source_hashes(tracked_files: tuple[str, ...] = TRACKED_SOURCE_FILES) -> dict[str, str]:
    return {
        relative_path: _file_sha256(resolve_project_path(relative_path))
        for relative_path in tracked_files
    }


# full snapshot of the code's state right now (commit + dirty flag + per-file hashes)
def build_code_state() -> dict[str, Any]:
    return {
        "git_commit": current_git_commit(),
        "git_dirty": current_git_dirty(),
        "source_hashes": current_source_hashes(),
    }


def code_state_mismatches(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Human-readable list of what drifted; empty if the code states match."""
    # confronta due "fotografie" (quella al momento della registrazione e quella di adesso)
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

    # loads the file if it's already there, otherwise starts with an empty registry
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._entries = json.loads(self.path.read_text())

    # aggiunge una nuova entry per uno sweep appena creato, con relativa fotografia del codice
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

    # filtra solo gli sweep che riguardano gli algoritmi passati (serve al comando worker)
    def entries_for_algorithms(self, algorithms: list[str]) -> dict[str, dict[str, Any]]:
        return {
            sweep_id: entry
            for sweep_id, entry in self._entries.items()
            if entry["algorithm"] in algorithms
        }

    # scrive tutto su disco in json, formattato leggibile
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2, sort_keys=True))
