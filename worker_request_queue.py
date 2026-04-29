from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def _utcnow() -> datetime:
    return datetime.now(UTC)


class WorkerRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class _QueuedRequest:
    request_id: str
    worker_id: str
    payload: dict[str, Any]
    created_at: datetime
    claimed_at: datetime | None = None
    claim_expires_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    response_body: dict[str, Any] | None = None
    error: str | None = None
    error_status_code: int | None = None
    completion_event: asyncio.Event = field(default_factory=asyncio.Event)


class WorkerRequestQueue:
    def __init__(self, *, lease_seconds: float = 120.0) -> None:
        self._lease_seconds = max(5.0, float(lease_seconds))
        self._lock = asyncio.Lock()
        self._requests: dict[str, _QueuedRequest] = {}

    async def enqueue_and_wait(
        self,
        *,
        worker_id: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> tuple[str, dict[str, Any]]:
        record = _QueuedRequest(
            request_id=uuid4().hex,
            worker_id=worker_id,
            payload=dict(payload),
            created_at=_utcnow(),
        )
        async with self._lock:
            self._requests[record.request_id] = record
        try:
            await asyncio.wait_for(record.completion_event.wait(), timeout=max(1.0, timeout_seconds))
        except asyncio.TimeoutError as exc:
            async with self._lock:
                current = self._requests.get(record.request_id)
                if current is not None and current.completed_at is None and current.failed_at is None:
                    del self._requests[record.request_id]
            raise TimeoutError(
                f"Timed out waiting for worker {worker_id} to complete request {record.request_id}"
            ) from exc
        if record.response_body is not None:
            return record.request_id, record.response_body
        raise WorkerRequestError(
            record.error or f"Worker {worker_id} failed request {record.request_id}",
            status_code=record.error_status_code,
        )

    async def poll(self, *, worker_id: str, limit: int = 1) -> list[dict[str, Any]]:
        now = _utcnow()
        ready: list[_QueuedRequest] = []
        async with self._lock:
            for record in self._requests.values():
                if record.worker_id != worker_id:
                    continue
                if record.completed_at is not None or record.failed_at is not None:
                    continue
                if record.claim_expires_at is not None and record.claim_expires_at > now:
                    continue
                record.claimed_at = now
                record.claim_expires_at = datetime.fromtimestamp(
                    now.timestamp() + self._lease_seconds,
                    tz=UTC,
                )
                ready.append(record)
                if len(ready) >= max(1, limit):
                    break
        return [
            {
                "request_id": item.request_id,
                "payload": item.payload,
                "created_at": item.created_at.isoformat(),
            }
            for item in ready
        ]

    async def complete(
        self,
        *,
        worker_id: str,
        request_id: str,
        response_body: dict[str, Any],
    ) -> None:
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None or record.worker_id != worker_id:
                raise KeyError(request_id)
            record.completed_at = _utcnow()
            record.response_body = dict(response_body)
            record.error = None
            record.error_status_code = None
            record.completion_event.set()

    async def fail(
        self,
        *,
        worker_id: str,
        request_id: str,
        error: str,
        status_code: int | None = None,
    ) -> None:
        async with self._lock:
            record = self._requests.get(request_id)
            if record is None or record.worker_id != worker_id:
                raise KeyError(request_id)
            record.failed_at = _utcnow()
            record.error = str(error).strip() or "Worker request failed"
            record.error_status_code = status_code
            record.completion_event.set()

    async def pending_count(self, *, worker_id: str) -> int:
        async with self._lock:
            return sum(
                1
                for record in self._requests.values()
                if record.worker_id == worker_id
                and record.completed_at is None
                and record.failed_at is None
            )
