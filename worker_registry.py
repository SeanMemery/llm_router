from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dashboard.llm_router import LLMRouterConnectionConfig


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


class LLMWorkerStatus(str, Enum):
    online = "online"
    offline = "offline"


class LLMWorkerTransportMode(str, Enum):
    direct_endpoint = "direct-endpoint"
    pull_request = "pull-request"


class LLMWorkerRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str | None = None
    display_name: str
    hostname: str
    endpoint_base_url: str | None = None
    model: str
    local_model_path: str | None = None
    local_model_root: str | None = None
    api_key: str | None = None
    max_concurrent_requests: int = Field(default=1, ge=1)
    supports_image_inputs: bool = False
    transport_mode: LLMWorkerTransportMode = LLMWorkerTransportMode.direct_endpoint
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fields(self) -> "LLMWorkerRegisterRequest":
        self.worker_id = _clean_optional(self.worker_id)
        self.display_name = self.display_name.strip()
        self.hostname = self.hostname.strip()
        self.endpoint_base_url = _clean_optional(self.endpoint_base_url)
        if self.endpoint_base_url is not None:
            self.endpoint_base_url = self.endpoint_base_url.rstrip("/")
        self.model = self.model.strip()
        self.local_model_path = _clean_optional(self.local_model_path)
        self.local_model_root = _clean_optional(self.local_model_root)
        self.api_key = _clean_optional(self.api_key)
        if not self.display_name:
            raise ValueError("display_name must be non-empty")
        if not self.hostname:
            raise ValueError("hostname must be non-empty")
        if self.transport_mode == LLMWorkerTransportMode.direct_endpoint and not self.endpoint_base_url:
            raise ValueError("endpoint_base_url must be non-empty for direct-endpoint workers")
        if not self.model:
            raise ValueError("model must be non-empty")
        return self


class LLMWorkerHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint_base_url: str | None = None
    model: str | None = None
    local_model_path: str | None = None
    local_model_root: str | None = None
    api_key: str | None = None
    max_concurrent_requests: int | None = Field(default=None, ge=1)
    supports_image_inputs: bool | None = None
    transport_mode: LLMWorkerTransportMode | None = None
    last_error: str | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "LLMWorkerHeartbeatRequest":
        self.endpoint_base_url = _clean_optional(self.endpoint_base_url)
        if self.endpoint_base_url is not None:
            self.endpoint_base_url = self.endpoint_base_url.rstrip("/")
        self.model = _clean_optional(self.model)
        self.local_model_path = _clean_optional(self.local_model_path)
        self.local_model_root = _clean_optional(self.local_model_root)
        self.api_key = _clean_optional(self.api_key)
        self.last_error = _clean_optional(self.last_error)
        return self


class LLMWorkerDisconnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "LLMWorkerDisconnectRequest":
        self.reason = _clean_optional(self.reason)
        return self


class LLMWorkerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str
    display_name: str
    hostname: str
    endpoint_base_url: str | None = None
    model: str
    local_model_path: str | None = None
    local_model_root: str | None = None
    api_key: str | None = None
    max_concurrent_requests: int = Field(ge=1)
    router_max_concurrent_requests: int | None = Field(default=None, ge=1)
    router_priority: int = Field(default=2, ge=1, le=99)
    supports_image_inputs: bool = False
    transport_mode: LLMWorkerTransportMode = LLMWorkerTransportMode.direct_endpoint
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: LLMWorkerStatus = LLMWorkerStatus.online
    registered_at: datetime
    last_heartbeat_at: datetime
    disconnected_at: datetime | None = None
    last_error: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "LLMWorkerRecord":
        self.worker_id = self.worker_id.strip()
        self.display_name = self.display_name.strip()
        self.hostname = self.hostname.strip()
        self.endpoint_base_url = _clean_optional(self.endpoint_base_url)
        if self.endpoint_base_url is not None:
            self.endpoint_base_url = self.endpoint_base_url.rstrip("/")
        self.model = self.model.strip()
        self.local_model_path = _clean_optional(self.local_model_path)
        self.local_model_root = _clean_optional(self.local_model_root)
        self.api_key = _clean_optional(self.api_key)
        self.last_error = _clean_optional(self.last_error)
        if not self.worker_id:
            raise ValueError("worker_id must be non-empty")
        if not self.display_name:
            raise ValueError("display_name must be non-empty")
        if not self.hostname:
            raise ValueError("hostname must be non-empty")
        if self.transport_mode == LLMWorkerTransportMode.direct_endpoint and not self.endpoint_base_url:
            raise ValueError("endpoint_base_url must be non-empty for direct-endpoint workers")
        if not self.model:
            raise ValueError("model must be non-empty")
        return self

    @property
    def effective_max_concurrent_requests(self) -> int:
        if self.router_max_concurrent_requests is None:
            return self.max_concurrent_requests
        return max(1, min(self.max_concurrent_requests, self.router_max_concurrent_requests))

    def to_connection_config(self) -> LLMRouterConnectionConfig:
        return LLMRouterConnectionConfig(
            connection_id=f"worker:{self.worker_id}",
            base_url=self.endpoint_base_url or f"worker://{self.worker_id}",
            model=self.model,
            source_kind="worker",
            worker_id=self.worker_id,
            api_key=self.api_key,
            starred=False,
            priority=self.router_priority,
            max_concurrent_requests=self.effective_max_concurrent_requests,
            supports_image_inputs=self.supports_image_inputs,
            transport_mode=self.transport_mode.value,
        )


class LLMWorkerRegistry:
    def __init__(
        self,
        *,
        workers: list[LLMWorkerRecord] | None = None,
        heartbeat_timeout_seconds: float = 90.0,
    ) -> None:
        self._workers: dict[str, LLMWorkerRecord] = {
            item.worker_id: item.model_copy(deep=True) for item in (workers or [])
        }
        self._heartbeat_timeout_seconds = max(5.0, float(heartbeat_timeout_seconds))

    @staticmethod
    def _sort_key(record: LLMWorkerRecord) -> tuple[int, str, str, str]:
        return (
            int(record.router_priority),
            (record.display_name or "").casefold(),
            (record.model or "").casefold(),
            record.worker_id,
        )

    def list_workers(self) -> list[LLMWorkerRecord]:
        return sorted(
            (item.model_copy(deep=True) for item in self._workers.values()),
            key=self._sort_key,
        )

    def get_worker(self, worker_id: str) -> LLMWorkerRecord:
        try:
            return self._workers[worker_id].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"Unknown worker {worker_id}") from exc

    def register(self, payload: LLMWorkerRegisterRequest) -> LLMWorkerRecord:
        now = _utcnow()
        worker_id = payload.worker_id or uuid4().hex
        existing = self._workers.get(worker_id)
        record = LLMWorkerRecord(
            worker_id=worker_id,
            display_name=payload.display_name,
            hostname=payload.hostname,
            endpoint_base_url=payload.endpoint_base_url,
            model=payload.model,
            local_model_path=payload.local_model_path,
            local_model_root=payload.local_model_root,
            api_key=payload.api_key,
            max_concurrent_requests=payload.max_concurrent_requests,
            router_max_concurrent_requests=(
                existing.router_max_concurrent_requests if existing is not None else None
            ),
            router_priority=(existing.router_priority if existing is not None else 2),
            supports_image_inputs=payload.supports_image_inputs,
            transport_mode=payload.transport_mode,
            metadata=dict(payload.metadata),
            status=LLMWorkerStatus.online,
            registered_at=existing.registered_at if existing is not None else now,
            last_heartbeat_at=now,
            disconnected_at=None,
            last_error=None,
        )
        self._workers[worker_id] = record
        return record.model_copy(deep=True)

    def heartbeat(self, worker_id: str, payload: LLMWorkerHeartbeatRequest) -> LLMWorkerRecord:
        record = self._workers.get(worker_id)
        if record is None:
            raise KeyError(f"Unknown worker {worker_id}")
        update: dict[str, Any] = {
            "status": LLMWorkerStatus.online,
            "last_heartbeat_at": _utcnow(),
            "disconnected_at": None,
        }
        if payload.endpoint_base_url is not None:
            update["endpoint_base_url"] = payload.endpoint_base_url
        if payload.model is not None:
            update["model"] = payload.model
        if payload.local_model_path is not None:
            update["local_model_path"] = payload.local_model_path
        if payload.local_model_root is not None:
            update["local_model_root"] = payload.local_model_root
        if payload.api_key is not None:
            update["api_key"] = payload.api_key
        if payload.max_concurrent_requests is not None:
            update["max_concurrent_requests"] = payload.max_concurrent_requests
        if payload.supports_image_inputs is not None:
            update["supports_image_inputs"] = payload.supports_image_inputs
        if payload.transport_mode is not None:
            update["transport_mode"] = payload.transport_mode
        if payload.metadata is not None:
            merged = dict(record.metadata)
            merged.update(payload.metadata)
            update["metadata"] = merged
        update["last_error"] = payload.last_error
        next_record = record.model_copy(update=update)
        self._workers[worker_id] = next_record
        return next_record.model_copy(deep=True)

    def set_router_max_concurrent_requests(
        self,
        worker_id: str,
        *,
        router_max_concurrent_requests: int | None,
    ) -> LLMWorkerRecord:
        record = self._workers.get(worker_id)
        if record is None:
            raise KeyError(f"Unknown worker {worker_id}")
        next_record = record.model_copy(
            update={"router_max_concurrent_requests": router_max_concurrent_requests}
        )
        self._workers[worker_id] = next_record
        return next_record.model_copy(deep=True)

    def set_router_priority(
        self,
        worker_id: str,
        *,
        router_priority: int,
    ) -> LLMWorkerRecord:
        record = self._workers.get(worker_id)
        if record is None:
            raise KeyError(f"Unknown worker {worker_id}")
        next_record = record.model_copy(update={"router_priority": router_priority})
        self._workers[worker_id] = next_record
        return next_record.model_copy(deep=True)

    def disconnect(self, worker_id: str, *, reason: str | None = None) -> LLMWorkerRecord:
        record = self._workers.get(worker_id)
        if record is None:
            raise KeyError(f"Unknown worker {worker_id}")
        now = _utcnow()
        next_record = record.model_copy(
            update={
                "status": LLMWorkerStatus.offline,
                "disconnected_at": now,
                "last_heartbeat_at": now,
                "last_error": reason,
            }
        )
        self._workers[worker_id] = next_record
        return next_record.model_copy(deep=True)

    def prune_stale_workers(self) -> list[LLMWorkerRecord]:
        now = _utcnow()
        stale_workers: list[LLMWorkerRecord] = []
        for worker_id, record in list(self._workers.items()):
            if record.status != LLMWorkerStatus.online:
                continue
            age_seconds = (now - record.last_heartbeat_at).total_seconds()
            if age_seconds < self._heartbeat_timeout_seconds:
                continue
            stale_record = record.model_copy(
                update={
                    "status": LLMWorkerStatus.offline,
                    "disconnected_at": now,
                    "last_error": f"Heartbeat timed out after {int(self._heartbeat_timeout_seconds)}s.",
                }
            )
            self._workers[worker_id] = stale_record
            stale_workers.append(stale_record.model_copy(deep=True))
        return stale_workers

    def active_connection_configs(self) -> list[LLMRouterConnectionConfig]:
        active_workers = [
            item
            for item in self._workers.values()
            if item.status == LLMWorkerStatus.online
        ]
        return [item.to_connection_config() for item in sorted(active_workers, key=self._sort_key)]
