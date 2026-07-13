from pathlib import Path


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    path = Path(path)

    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[1]
    return project_root / path
