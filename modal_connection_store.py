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


class ModalConnectionStatus(str, Enum):
    starting = "starting"
    running = "running"
    stopping = "stopping"
    stopped = "stopped"
    expired = "expired"
    failed = "failed"


class ModalConnectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    launch_id: str
    display_name: str
    model: str
    served_model_name: str
    gpu: str
    duration_minutes: int = Field(ge=1, le=10080)
    started_at: datetime = Field(default_factory=_utcnow)
    expires_at: datetime
    pid: int | None = Field(default=None, ge=1)
    app_id: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    log_path: str
    state_path: str
    status: ModalConnectionStatus = ModalConnectionStatus.starting
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("launch_id", "display_name", "model", "served_model_name", "gpu", "log_path", "state_path")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must be non-empty")
        return text

    @field_validator("app_id", "base_url", "api_key", "last_error")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None


def modal_connection_store_path(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "modal_connections.json"


def load_modal_connections(layout: ArtifactLayout) -> list[ModalConnectionRecord]:
    path = modal_connection_store_path(layout)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Malformed modal connections store: {path}")
    return [ModalConnectionRecord.model_validate(item) for item in payload if isinstance(item, dict)]


def save_modal_connections(
    layout: ArtifactLayout,
    records: list[ModalConnectionRecord],
) -> Path:
    path = modal_connection_store_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([item.model_dump(mode="json") for item in records], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
