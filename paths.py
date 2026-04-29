from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    project_root: Path
    config_root: Path
    environment_manifest_root: Path
    data_root: Path
    evaluations_root: Path
    runs_root: Path
    reports_root: Path


def get_project_root() -> Path:
    configured = os.environ.get("PATTERN_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "configs").is_dir() and (cwd / "data").is_dir():
        return cwd
    return Path("/home/sean/Nextcloud/PhD/Year_4/pattern-learning-v2").resolve()


def get_artifact_layout() -> ArtifactLayout:
    project_root = get_project_root()
    config_root = project_root / "configs"
    return ArtifactLayout(
        project_root=project_root,
        config_root=config_root,
        environment_manifest_root=config_root,
        data_root=project_root / "data",
        evaluations_root=project_root / "evaluations",
        runs_root=project_root / "runs",
        reports_root=project_root / "reports",
    )
