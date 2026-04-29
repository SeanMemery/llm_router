from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import petname

from paths import get_artifact_layout


PATTERN_LIBRARY_FILENAME = "pattern_library.json"
UNSUPERVISED_RUNS_DIRNAME = "unsupervised"
_ISSUED_RUN_IDS: set[str] = set()


def _generated_run_id() -> str:
    for _ in range(32):
        timestamp = datetime.now(timezone.utc).strftime("%m-%d-%H-%M-%S")
        noun = str(petname.generate(words=1)).strip().lower().replace(" ", "-")
        if not noun:
            continue
        run_id = f"{timestamp}-{noun}"
        if run_id in _ISSUED_RUN_IDS:
            continue
        _ISSUED_RUN_IDS.add(run_id)
        return run_id
    raise RuntimeError("Failed to generate a unique pattern run id")


def default_pattern_run_dir(env_id: str, *, timestamp: str | None = None) -> Path:
    layout = get_artifact_layout()
    run_id = timestamp or _generated_run_id()
    return layout.data_root / "patterns" / env_id / run_id


def pattern_library_path_for_run_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / PATTERN_LIBRARY_FILENAME


def default_pattern_output_path(env_id: str, *, timestamp: str | None = None) -> Path:
    return pattern_library_path_for_run_dir(default_pattern_run_dir(env_id, timestamp=timestamp))


def resolve_pattern_run_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate
    if candidate.is_file():
        return candidate.parent
    if candidate.suffix == ".json":
        return candidate.parent
    return candidate


def resolve_pattern_library_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return pattern_library_path_for_run_dir(candidate)
    if candidate.suffix == ".json":
        return candidate
    return pattern_library_path_for_run_dir(candidate)


def default_pattern_evaluation_root(*, pattern_run_dir: str | Path, evaluation_run_id: str) -> Path:
    return Path(pattern_run_dir) / "deepphy_eval" / evaluation_run_id


def default_unsupervised_pattern_root(env_id: str) -> Path:
    layout = get_artifact_layout()
    return layout.data_root / "patterns" / env_id / UNSUPERVISED_RUNS_DIRNAME


def default_unsupervised_run_dir(env_id: str, *, timestamp: str | None = None) -> Path:
    run_id = timestamp or _generated_run_id()
    return default_unsupervised_pattern_root(env_id) / run_id


def is_unsupervised_run_dir(path: Path) -> bool:
    candidate = Path(path)
    return candidate.parent.name == UNSUPERVISED_RUNS_DIRNAME
