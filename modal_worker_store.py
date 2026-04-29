from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from paths import ArtifactLayout


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ModalWorkerLaunchStatus(str, Enum):
    starting = "starting"
    running = "running"
    stopped = "stopped"
    expired = "expired"
    exited = "exited"
    failed = "failed"


class ModalWorkerLaunchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    launch_id: str
    display_name: str
    model: str
    duration_minutes: int = Field(ge=1, le=10080)
    started_at: datetime
    expires_at: datetime
    pid: int | None = Field(default=None, ge=1)
    worker_id: str | None = None
    log_path: str
    command: list[str] = Field(default_factory=list)
    status: ModalWorkerLaunchStatus = ModalWorkerLaunchStatus.starting
    exit_code: int | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("launch_id", "display_name", "model", "log_path")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must be non-empty")
        return text

    @field_validator("worker_id", "last_error")
    @classmethod
    def _clean_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None


def modal_worker_store_path(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "modal_worker_launches.json"


def load_modal_worker_launches(layout: ArtifactLayout) -> list[ModalWorkerLaunchRecord]:
    path = modal_worker_store_path(layout)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Malformed modal worker store: {path}")
    return [ModalWorkerLaunchRecord.model_validate(item) for item in payload if isinstance(item, dict)]


def save_modal_worker_launches(
    layout: ArtifactLayout,
    launches: list[ModalWorkerLaunchRecord],
) -> Path:
    path = modal_worker_store_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([item.model_dump(mode="json") for item in launches], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
