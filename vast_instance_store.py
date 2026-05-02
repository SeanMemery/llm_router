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


class VastInstanceStatus(str, Enum):
    starting = "starting"
    running = "running"
    stopping = "stopping"
    stopped = "stopped"
    missing = "missing"
    failed = "failed"


class VastInstanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "vast"
    instance_id: int | None = Field(default=None, ge=1)
    offer_id: int | None = Field(default=None, ge=1)
    label: str
    model: str
    served_model_name: str
    endpoint_port: int = Field(default=8000, ge=1, le=65535)
    base_url: str | None = None
    api_key: str | None = None
    public_ipaddr: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = Field(default=None, ge=1, le=65535)
    direct_ssh_host: str | None = None
    direct_ssh_port: int | None = Field(default=None, ge=1, le=65535)
    status: VastInstanceStatus = VastInstanceStatus.starting
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("label", "model", "served_model_name")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must be non-empty")
        return text

    @field_validator("base_url", "api_key", "public_ipaddr", "ssh_host", "direct_ssh_host", "last_error")
    @classmethod
    def _clean_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None


def vast_instance_store_path(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "llm_router_vast_instance.json"


def load_vast_instance(layout: ArtifactLayout) -> VastInstanceRecord | None:
    path = vast_instance_store_path(layout)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload is None:
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed Vast instance store: {path}")
    return VastInstanceRecord.model_validate(payload)


def save_vast_instance(layout: ArtifactLayout, record: VastInstanceRecord | None) -> Path:
    path = vast_instance_store_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    if record is None:
        path.unlink(missing_ok=True)
        return path
    path.write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
