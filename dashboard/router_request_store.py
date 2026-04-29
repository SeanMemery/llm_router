from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _compact_metric(value: int | None) -> str:
    if value is None:
        return "--"
    abs_value = abs(int(value))
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _timestamp_label(value: str | None) -> str:
    if not value:
        return "Unknown time"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _duration_label(value: float | None) -> str:
    if value is None:
        return "--"
    if value >= 10.0:
        return f"{value:.1f}s"
    if value >= 1.0:
        return f"{value:.2f}s"
    return f"{value * 1000:.0f}ms"


def _base_url_display(value: str | None) -> str:
    if not value:
        return "Unknown route"
    parsed = urlparse(value)
    if parsed.netloc:
        suffix = parsed.path.rstrip("/")
        return f"{parsed.netloc}{suffix}" if suffix else parsed.netloc
    return value


class RouterRequestStore:
    def __init__(
        self,
        *,
        index_path: Path,
        detail_dir: Path,
        max_entries: int = 1000,
    ) -> None:
        self.index_path = index_path
        self.detail_dir = detail_dir
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._entries: list[dict[str, Any]] = []
        self._cached_payload: dict[str, Any] | None = None
        self._load()

    def _load(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        entries = payload.get("requests")
        if not isinstance(entries, list):
            return
        loaded: list[dict[str, Any]] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            request_id = str(item.get("id") or "").strip()
            timestamp = str(item.get("timestamp") or "").strip()
            status = str(item.get("status") or "").strip() or "failed"
            if not request_id:
                continue
            loaded.append(
                {
                    "id": request_id,
                    "timestamp": timestamp,
                    "timestamp_label": str(item.get("timestamp_label") or _timestamp_label(timestamp)),
                    "status": status,
                    "status_label": str(
                        item.get("status_label")
                        or ("Completed" if status == "completed" else "Failed")
                    ),
                    "requested_model": item.get("requested_model"),
                    "selected_model": item.get("selected_model"),
                    "base_url": item.get("base_url"),
                    "base_url_display": item.get("base_url_display")
                    or _base_url_display(item.get("base_url")),
                    "prompt_tokens": item.get("prompt_tokens"),
                    "completion_tokens": item.get("completion_tokens"),
                    "prompt_tokens_label": str(
                        item.get("prompt_tokens_label") or _compact_metric(item.get("prompt_tokens"))
                    ),
                    "completion_tokens_label": str(
                        item.get("completion_tokens_label")
                        or _compact_metric(item.get("completion_tokens"))
                    ),
                    "duration_seconds": item.get("duration_seconds"),
                    "duration_label": str(
                        item.get("duration_label") or _duration_label(item.get("duration_seconds"))
                    ),
                    "http_status_code": item.get("http_status_code"),
                    "source_kind": item.get("source_kind"),
                    "transport_mode": item.get("transport_mode"),
                    "worker_id": item.get("worker_id"),
                }
            )
        self._entries = loaded[: self.max_entries]

    def _detail_path(self, request_id: str) -> Path:
        return self.detail_dir / f"{request_id}.json"

    def _persist_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_entries": self.max_entries,
            "total_retained": len(self._entries),
            "requests": self._entries,
        }
        self.index_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _summarize(self, entry: dict[str, Any]) -> dict[str, Any]:
        usage = entry.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        connection = entry.get("connection")
        if not isinstance(connection, dict):
            connection = {}
        timing = entry.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(prompt_tokens, int):
            prompt_tokens = None
        if not isinstance(completion_tokens, int):
            completion_tokens = None
        status = str(entry.get("status") or "").strip() or "failed"
        timestamp = str(entry.get("completed_at") or entry.get("started_at") or "").strip()
        duration_seconds = timing.get("router_request_seconds")
        if not isinstance(duration_seconds, (int, float)):
            duration_seconds = None
        return {
            "id": str(entry.get("id") or "").strip(),
            "timestamp": timestamp,
            "timestamp_label": _timestamp_label(timestamp),
            "status": status,
            "status_label": "Completed" if status == "completed" else "Failed",
            "requested_model": entry.get("requested_model"),
            "selected_model": connection.get("selected_model"),
            "base_url": connection.get("base_url"),
            "base_url_display": _base_url_display(connection.get("base_url")),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens_label": _compact_metric(prompt_tokens),
            "completion_tokens_label": _compact_metric(completion_tokens),
            "duration_seconds": round(float(duration_seconds), 6) if duration_seconds is not None else None,
            "duration_label": _duration_label(float(duration_seconds)) if duration_seconds is not None else "--",
            "http_status_code": entry.get("http_status_code"),
            "source_kind": connection.get("source_kind"),
            "transport_mode": connection.get("transport_mode"),
            "worker_id": connection.get("worker_id"),
        }

    def record(self, entry: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            request_id = str(entry.get("id") or "").strip()
            if not request_id:
                raise ValueError("request log entry must include a non-empty id")
            detail = dict(entry)
            summary = self._summarize(detail)
            detail["summary"] = summary
            self.detail_dir.mkdir(parents=True, exist_ok=True)
            self._detail_path(request_id).write_text(
                json.dumps(detail, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._entries = [summary, *[item for item in self._entries if item.get("id") != request_id]]
            stale = self._entries[self.max_entries :]
            self._entries = self._entries[: self.max_entries]
            for item in stale:
                stale_id = str(item.get("id") or "").strip()
                if not stale_id:
                    continue
                try:
                    self._detail_path(stale_id).unlink()
                except FileNotFoundError:
                    pass
            self._cached_payload = None
            self._persist_index()
            return summary

    def payload(self, *, limit: int | None = None) -> dict[str, Any]:
        with self._lock:
            if limit is None and self._cached_payload is not None:
                return self._cached_payload
            requests = self._entries
            if limit is not None:
                requests = requests[: max(0, int(limit))]
            payload = {
                "max_entries": self.max_entries,
                "total_retained": len(self._entries),
                "requests": requests,
            }
            if limit is None:
                self._cached_payload = payload
            return payload

    def detail(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            request_id = request_id.strip()
            if not request_id:
                return None
            path = self._detail_path(request_id)
            if not path.is_file():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
            if not isinstance(payload, dict):
                return None
            return payload
