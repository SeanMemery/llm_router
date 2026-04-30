from __future__ import annotations

import asyncio
import hmac
import hashlib
import importlib.util
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
from uuid import uuid4

import httpx
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field, ValidationError

from dashboard.llm_router import (
    DashboardLLMRouter,
    LLMRouterConnectionConfig,
    LLMRouterConnectionSnapshot,
    LLMRouterSnapshot,
    NoActiveLLMConnectionError,
    UnknownLLMModelError,
)
from dashboard.llm_router_store import load_llm_router_snapshot, save_llm_router_snapshot
from dashboard.router_request_store import RouterRequestStore
from env_utils import load_dotenv_file
from llm_config import load_shared_llm_config
from modal_worker_store import (
    ModalWorkerLaunchRecord,
    ModalWorkerLaunchStatus,
    load_modal_worker_launches,
    save_modal_worker_launches,
)
from paths import ArtifactLayout, get_artifact_layout
from vast_instance_store import (
    VastInstanceRecord,
    VastInstanceStatus,
    load_vast_instance,
    save_vast_instance,
)
from worker_registry import (
    LLMWorkerDisconnectRequest,
    LLMWorkerHeartbeatRequest,
    LLMWorkerRecord,
    LLMWorkerRegisterRequest,
    LLMWorkerRegistry,
)
from worker_request_queue import WorkerRequestError, WorkerRequestQueue
from worker_store import load_llm_workers, save_llm_workers

load_dotenv_file(Path(__file__).resolve().parent / ".env")

ROUTER_HEALTHCHECK_INTERVAL_SECONDS = 1.0
ROUTER_WORKER_HEARTBEAT_TIMEOUT_SECONDS = max(
    30.0,
    float(os.getenv("ROUTER_WORKER_HEARTBEAT_TIMEOUT_SECONDS", "180")),
)
ROUTER_PUBLIC_FUNNEL_PORT = 8443
ROUTER_PUBLIC_FUNNEL_PATH = "/public"
ROUTER_PUBLIC_API_PATH = f"{ROUTER_PUBLIC_FUNNEL_PATH}/v1"
ROUTER_PUBLIC_API_KEY_HEADER = "Bearer"
ROUTER_METRICS_WINDOW_SECONDS = 3600
ROUTER_METRICS_SAMPLE_INTERVAL_SECONDS = max(
    5.0,
    float(os.getenv("ROUTER_METRICS_SAMPLE_INTERVAL_SECONDS", "10")),
)
ROUTER_REQUEST_LOG_MAX_ENTRIES = max(
    100,
    int(os.getenv("ROUTER_REQUEST_LOG_MAX_ENTRIES", "1000")),
)
DEFAULT_MODAL_WORKER_MODEL = "RedHatAI/Qwen3.6-35B-A3B-NVFP4"
DEFAULT_MODAL_WORKER_MODELS = (
    DEFAULT_MODAL_WORKER_MODEL,
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "Qwen/Qwen2.5-32B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
)
DEFAULT_VAST_MODEL = "RedHatAI/Qwen3.6-35B-A3B-NVFP4"
DEFAULT_VAST_SERVED_MODEL_NAME = "Qwen3.6-35B-A3B-UD-Q4_K_M"
DEFAULT_VAST_INSTANCE_LABEL = "llm-router-vast-qwen36"
DEFAULT_VAST_IMAGE = "vllm/vllm-openai:latest"
DEFAULT_VAST_ENDPOINT_PORT = 8000
DEFAULT_VAST_DISK_GB = 120
DEFAULT_VAST_MAX_MODEL_LEN = 2048
DEFAULT_VAST_GPU_MEMORY_UTILIZATION = 0.72
DEFAULT_VAST_MAX_CONCURRENT = 8
DEFAULT_VAST_ENDPOINT_PRIORITY = 1
DEFAULT_VAST_REASONING_PARSER = "qwen3"
DEFAULT_VAST_MOE_BACKEND = "flashinfer_cutlass"
DEFAULT_VAST_STARTUP_GRACE_SECONDS = 900
DEFAULT_VAST_SEARCH_QUERY = (
    "reliability > 0.99 rented=False rentable=True verified=True "
    "gpu_name=RTX_3090 num_gpus=1 disk_space>120 direct_port_count>=8 "
    "geolocation in [US,CA,GB,DE,NL,FR,BE,ES,PL]"
)
DEFAULT_VAST_SEARCH_ORDER = "dph_total"
DEFAULT_VAST_DENYLISTED_HOST_IDS = {44511, 155125, 51981}
DEFAULT_VAST_PREFERRED_HOST_IDS = {228989, 148689, 51981, 67245}
DEFAULT_VAST_MANAGED_BASE_URLS = {
    "http://209.146.116.50:36413",
    "http://209.146.116.50:36468",
}
MODAL_APP_STATUS_CACHE_TTL_SECONDS = max(
    5.0,
    float(os.getenv("MODAL_APP_STATUS_CACHE_TTL_SECONDS", "15")),
)
_MODAL_APP_STATUS_CACHE_LOCK = threading.RLock()
_MODAL_APP_STATUS_CACHE_EXPIRES_AT = 0.0
_MODAL_APP_STATUS_CACHE: dict[str, str] = {}


class RouterMetricsHistoryStore:
    def __init__(
        self,
        path: Path,
        *,
        window_seconds: int = ROUTER_METRICS_WINDOW_SECONDS,
        sample_interval_seconds: float = ROUTER_METRICS_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self.path = path
        self.window_seconds = max(60, int(window_seconds))
        self.sample_interval_seconds = max(1.0, float(sample_interval_seconds))
        self._lock = threading.RLock()
        self._points: list[dict[str, Any]] = []
        self._last_sample_epoch: float | None = None
        self._cached_payload: dict[str, Any] | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        points = payload.get("points") if isinstance(payload, dict) else None
        if not isinstance(points, list):
            return
        loaded: list[dict[str, Any]] = []
        for item in points:
            if not isinstance(item, dict):
                continue
            try:
                epoch_seconds = float(item.get("epoch_seconds") or 0.0)
                in_flight_requests = int(item.get("in_flight_requests") or 0)
                throughput = float(item.get("throughput") or 0.0)
            except (TypeError, ValueError):
                continue
            if epoch_seconds <= 0.0:
                continue
            loaded.append(
                {
                    "epoch_seconds": epoch_seconds,
                    "in_flight_requests": max(0, in_flight_requests),
                    "throughput": max(0.0, throughput),
                }
            )
        self._points = loaded[-max(1, int(self.window_seconds / self.sample_interval_seconds) * 2) :]
        if self._points:
            self._last_sample_epoch = float(self._points[-1]["epoch_seconds"])
        self._trim()

    def _trim(self) -> None:
        if not self._points:
            return
        cutoff = time.time() - float(self.window_seconds)
        self._points = [
            item for item in self._points if float(item.get("epoch_seconds") or 0.0) >= cutoff
        ]

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "window_seconds": self.window_seconds,
            "sample_interval_seconds": self.sample_interval_seconds,
            "points": self._points,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def record(self, *, in_flight_requests: int, throughput: float) -> bool:
        with self._lock:
            now = time.time()
            if self._last_sample_epoch is not None and (now - self._last_sample_epoch) < self.sample_interval_seconds:
                return False
            self._points.append(
                {
                    "epoch_seconds": now,
                    "in_flight_requests": max(0, int(in_flight_requests)),
                    "throughput": max(0.0, float(throughput)),
                }
            )
            self._last_sample_epoch = now
            self._trim()
            self._cached_payload = None
            self._persist()
            return True

    def payload(self) -> dict[str, Any]:
        with self._lock:
            if self._cached_payload is not None:
                return self._cached_payload
            points = [
                {
                    "timestamp": datetime.fromtimestamp(
                        float(item["epoch_seconds"]),
                        tz=timezone.utc,
                    ).isoformat(),
                    "epoch_seconds": round(float(item["epoch_seconds"]), 3),
                    "in_flight_requests": int(item["in_flight_requests"]),
                    "throughput": round(float(item["throughput"]), 3),
                }
                for item in self._points
            ]
            self._cached_payload = {
                "window_seconds": self.window_seconds,
                "sample_interval_seconds": self.sample_interval_seconds,
                "throughput_unit": "tokens_per_second",
                "points": points,
            }
            return self._cached_payload


class WorkerRequestPollRequest(BaseModel):
    max_requests: int = Field(default=1, ge=1, le=32)


class WorkerConcurrencySettingsRequest(BaseModel):
    max_concurrent_requests: int | None = Field(default=None, ge=1, le=32)


class RouterPrioritySettingsRequest(BaseModel):
    priority: int = Field(default=2, ge=1, le=99)


class WorkerRequestCompleteRequest(BaseModel):
    response_body: dict[str, Any]


class WorkerRequestFailRequest(BaseModel):
    error: str
    status_code: int | None = Field(default=None, ge=100, le=599)


def _coerce_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _router_env_file_path(layout: ArtifactLayout) -> Path:
    return layout.project_root / ".env"


def _router_control_base_url(value: str) -> str:
    base = str(value).strip().rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return base


def _modal_worker_model_options() -> list[str]:
    raw = os.getenv("MODAL_WORKER_MODEL_OPTIONS", "")
    configured = [
        item.strip()
        for item in raw.replace("\n", ",").split(",")
        if item.strip()
    ]
    ordered: list[str] = []
    seen: set[str] = set()
    for item in [*configured, *DEFAULT_MODAL_WORKER_MODELS]:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _modal_worker_log_path(layout: ArtifactLayout, launch_id: str) -> Path:
    return layout.data_root / "dashboard" / "modal_workers" / f"{launch_id}.log"


def _modal_worker_state_path(layout: ArtifactLayout, launch_id: str) -> Path:
    return layout.data_root / "dashboard" / "modal_workers" / f"{launch_id}.state.json"


def _load_modal_worker_state(layout: ArtifactLayout, launch_id: str) -> dict[str, Any]:
    path = _modal_worker_state_path(layout, launch_id)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _stop_modal_launch_app(*, layout: ArtifactLayout, launch_id: str, modal_app_id: str | None) -> tuple[bool, str | None]:
    app_id = _coerce_optional(modal_app_id)
    if app_id is None:
        state_payload = _load_modal_worker_state(layout, launch_id)
        app_id = _coerce_optional(state_payload.get("app_id"))
    if app_id is None:
        return True, None
    env = os.environ.copy()
    env.update(load_dotenv_file(_router_env_file_path(layout)))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "modal", "app", "stop", "-y", app_id],
            cwd=str(layout.project_root),
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=120.0,
        )
    except Exception as exc:  # noqa: BLE001
        return False, str(exc).strip() or exc.__class__.__name__
    if result.returncode == 0:
        return True, None
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    message = stderr or stdout or f"modal app stop exited with status {result.returncode}"
    return False, message


def _modal_app_state_map(layout: ArtifactLayout) -> dict[str, str]:
    global _MODAL_APP_STATUS_CACHE_EXPIRES_AT, _MODAL_APP_STATUS_CACHE
    with _MODAL_APP_STATUS_CACHE_LOCK:
        now = time.monotonic()
        if now < _MODAL_APP_STATUS_CACHE_EXPIRES_AT and _MODAL_APP_STATUS_CACHE:
            return dict(_MODAL_APP_STATUS_CACHE)
        env = os.environ.copy()
        env.update(load_dotenv_file(_router_env_file_path(layout)))
        try:
            result = subprocess.run(
                [sys.executable, "-m", "modal", "app", "list", "--json"],
                cwd=str(layout.project_root),
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
        except Exception:
            return dict(_MODAL_APP_STATUS_CACHE)
        if result.returncode != 0:
            return dict(_MODAL_APP_STATUS_CACHE)
        try:
            payload = json.loads(result.stdout or "[]")
        except Exception:
            return dict(_MODAL_APP_STATUS_CACHE)
        states: dict[str, str] = {}
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                app_id = _coerce_optional(item.get("App ID"))
                app_state = _coerce_optional(item.get("State"))
                if app_id is None or app_state is None:
                    continue
                states[app_id] = app_state.lower()
        _MODAL_APP_STATUS_CACHE = states
        _MODAL_APP_STATUS_CACHE_EXPIRES_AT = now + MODAL_APP_STATUS_CACHE_TTL_SECONDS
        return dict(states)


def _modal_app_state_is_live(state: str | None) -> bool:
    if state is None:
        return True
    normalized = str(state).strip().lower()
    return normalized not in {"stopped", "stopping", "failed", "error", "exited", "completed"}


def _prune_modal_worker_launch_files(layout: ArtifactLayout, launch_id: str) -> None:
    for path in (
        _modal_worker_log_path(layout, launch_id),
        _modal_worker_state_path(layout, launch_id),
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def _reconcile_modal_worker_launches(
    *,
    layout: ArtifactLayout,
    workers: list[LLMWorkerRecord],
) -> list[ModalWorkerLaunchRecord]:
    launches = load_modal_worker_launches(layout)
    worker_by_launch_id: dict[str, LLMWorkerRecord] = {}
    for worker in workers:
        if worker.status.value != "online":
            continue
        metadata = worker.metadata if isinstance(worker.metadata, dict) else {}
        if metadata.get("launcher") != "modal_worker":
            continue
        launch_id = _coerce_optional(metadata.get("launch_id"))
        if launch_id is None:
            continue
        worker_by_launch_id[launch_id] = worker
    modal_app_states = _modal_app_state_map(layout)
    now = datetime.now(timezone.utc)
    updated: list[ModalWorkerLaunchRecord] = []
    pruned_launch_ids: list[str] = []
    dirty = False
    for launch in launches:
        matched_worker = worker_by_launch_id.get(launch.launch_id)
        next_status = launch.status
        next_worker_id = launch.worker_id
        next_metadata = dict(launch.metadata)
        next_last_error = launch.last_error
        state_payload = _load_modal_worker_state(layout, launch.launch_id)
        state_app_id = _coerce_optional(state_payload.get("app_id"))
        state_worker_id = _coerce_optional(state_payload.get("worker_id"))
        state_execution_mode = _coerce_optional(state_payload.get("execution_mode"))
        state_value = _coerce_optional(state_payload.get("state"))
        state_last_error = _coerce_optional(state_payload.get("last_error"))
        if state_app_id is not None:
            next_metadata["modal_app_id"] = state_app_id
        if state_execution_mode is not None:
            next_metadata["execution_mode"] = state_execution_mode
        if state_last_error is not None:
            next_last_error = state_last_error
        modal_app_id = _coerce_optional(next_metadata.get("modal_app_id"))
        modal_app_state = modal_app_states.get(modal_app_id) if modal_app_id is not None else None
        if modal_app_state is not None:
            next_metadata["modal_app_state"] = modal_app_state
        if matched_worker is not None and isinstance(matched_worker.metadata, dict):
            worker_app_id = _coerce_optional(matched_worker.metadata.get("modal_app_id"))
            worker_execution_mode = _coerce_optional(matched_worker.metadata.get("execution_mode"))
            if worker_app_id is not None:
                next_metadata["modal_app_id"] = worker_app_id
            if worker_execution_mode is not None:
                next_metadata["execution_mode"] = worker_execution_mode
            modal_app_id = _coerce_optional(next_metadata.get("modal_app_id"))
            modal_app_state = modal_app_states.get(modal_app_id) if modal_app_id is not None else modal_app_state
            if modal_app_state is not None:
                next_metadata["modal_app_state"] = modal_app_state
        if not _modal_app_state_is_live(modal_app_state):
            next_worker_id = None
            next_status = ModalWorkerLaunchStatus.stopped
        elif matched_worker is not None:
            next_status = ModalWorkerLaunchStatus.running
            next_worker_id = matched_worker.worker_id
        elif launch.status in {ModalWorkerLaunchStatus.stopped, ModalWorkerLaunchStatus.failed}:
            next_worker_id = None
            next_status = launch.status
        elif state_value == "stopped":
            next_worker_id = None
            next_status = ModalWorkerLaunchStatus.stopped
        elif state_value == "failed":
            next_worker_id = None
            next_status = ModalWorkerLaunchStatus.failed
        elif state_worker_id is not None:
            next_worker_id = state_worker_id
            if state_value == "registered":
                next_status = ModalWorkerLaunchStatus.running
            elif state_value == "modal_running":
                next_status = ModalWorkerLaunchStatus.starting
        else:
            next_worker_id = None
            if _pid_is_alive(launch.pid):
                next_status = ModalWorkerLaunchStatus.starting
            elif now >= launch.expires_at:
                next_status = ModalWorkerLaunchStatus.expired
            else:
                next_status = ModalWorkerLaunchStatus.exited
        if (
            next_status != launch.status
            or next_worker_id != launch.worker_id
            or next_metadata != launch.metadata
            or next_last_error != launch.last_error
        ):
            dirty = True
            launch = launch.model_copy(
                update={
                    "status": next_status,
                    "worker_id": next_worker_id,
                    "metadata": next_metadata,
                    "last_error": next_last_error,
                }
            )
        if launch.status not in {ModalWorkerLaunchStatus.starting, ModalWorkerLaunchStatus.running}:
            dirty = True
            pruned_launch_ids.append(launch.launch_id)
            continue
        updated.append(launch)
    if dirty:
        save_modal_worker_launches(layout, updated)
    for launch_id in pruned_launch_ids:
        _prune_modal_worker_launch_files(layout, launch_id)
    return sorted(updated, key=lambda item: item.started_at, reverse=True)


def _run_vast_command(*args: str, timeout_seconds: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["vastai", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _parse_vast_json(result: subprocess.CompletedProcess[str]) -> Any:
    if result.returncode != 0:
        output = "\n".join(part for part in ((result.stdout or "").strip(), (result.stderr or "").strip()) if part)
        raise RuntimeError(output or f"vastai exited with status {result.returncode}")
    text = (result.stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse vastai JSON output: {exc}") from exc


def _vast_onstart_command(*, model: str, served_model_name: str, api_key: str) -> str:
    return (
        "bash -lc 'set -euo pipefail; "
        "mkdir -p /workspace/logs /workspace/hf-cache; "
        "export HF_HOME=/workspace/hf-cache; "
        "export HF_HUB_ENABLE_HF_TRANSFER=1; "
        "export CUDA_MODULE_LOADING=LAZY; "
        "vllm serve "
        f"{model} "
        "--host 0.0.0.0 "
        f"--port {DEFAULT_VAST_ENDPOINT_PORT} "
        f"--served-model-name {served_model_name} "
        f"--max-model-len {DEFAULT_VAST_MAX_MODEL_LEN} "
        f"--gpu-memory-utilization {DEFAULT_VAST_GPU_MEMORY_UTILIZATION:.2f} "
        f"--reasoning-parser {DEFAULT_VAST_REASONING_PARSER} "
        f"--moe-backend {DEFAULT_VAST_MOE_BACKEND} "
        "--enforce-eager "
        f"--api-key {api_key} "
        "2>&1 | tee /workspace/logs/vllm.log'"
    )


def _vast_endpoint_base_url(record: VastInstanceRecord) -> str | None:
    if not record.public_ipaddr:
        return None
    return f"http://{record.public_ipaddr}:{record.endpoint_port}"


def _vast_published_endpoint_port(payload: dict[str, Any], default_port: int) -> int:
    ports = payload.get("ports")
    if not isinstance(ports, dict):
        return default_port
    candidates = ports.get(f"{default_port}/tcp")
    if not isinstance(candidates, list):
        return default_port
    for item in candidates:
        if not isinstance(item, dict):
            continue
        raw_host_port = item.get("HostPort")
        if isinstance(raw_host_port, str) and raw_host_port.isdigit():
            return int(raw_host_port)
        if isinstance(raw_host_port, int | float):
            return int(raw_host_port)
    return default_port


def _known_vast_base_urls(record: VastInstanceRecord | None) -> set[str]:
    urls = set(DEFAULT_VAST_MANAGED_BASE_URLS)
    if record is None:
        return urls
    if isinstance(record.base_url, str) and record.base_url.strip():
        urls.add(record.base_url.strip())
    previous_base_urls = record.metadata.get("previous_base_urls")
    if isinstance(previous_base_urls, list):
        for item in previous_base_urls:
            if isinstance(item, str) and item.strip():
                urls.add(item.strip())
    return urls


def _probe_vast_endpoint(*, base_url: str, api_key: str | None, timeout_seconds: float = 6.0) -> str | None:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for path in ("/health", "/v1/models"):
        request = urllib.request.Request(f"{base_url}{path}", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if 200 <= response.status < 500:
                    return None
        except urllib.error.HTTPError as exc:
            if 200 <= exc.code < 500:
                return None
            return f"{path}: HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001
            return f"{path}: {exc}"
    return "No healthy Vast endpoint probe succeeded"


def _vast_instance_payload(record: VastInstanceRecord | None) -> dict[str, Any]:
    if record is None:
        return {
            "configured": False,
            "status": "missing",
            "status_label": "Not started",
            "status_class": "failed",
            "model": DEFAULT_VAST_MODEL,
            "served_model_name": DEFAULT_VAST_SERVED_MODEL_NAME,
            "endpoint_port": DEFAULT_VAST_ENDPOINT_PORT,
        }
    status_map = {
        VastInstanceStatus.running: ("Running", "running"),
        VastInstanceStatus.starting: ("Starting", "failed"),
        VastInstanceStatus.stopping: ("Stopping", "failed"),
        VastInstanceStatus.stopped: ("Stopped", "failed"),
        VastInstanceStatus.failed: ("Failed", "failed"),
        VastInstanceStatus.missing: ("Missing", "failed"),
    }
    status_label, status_class = status_map.get(record.status, ("Unknown", "failed"))
    ssh_command = None
    if record.direct_ssh_host and record.direct_ssh_port:
        ssh_command = f"ssh -p {record.direct_ssh_port} root@{record.direct_ssh_host}"
    elif record.ssh_host and record.ssh_port:
        ssh_command = f"ssh -p {record.ssh_port} root@{record.ssh_host}"
    return {
        "configured": True,
        "instance_id": record.instance_id,
        "offer_id": record.offer_id,
        "label": record.label,
        "status": record.status.value,
        "status_label": status_label,
        "status_class": status_class,
        "model": record.model,
        "served_model_name": record.served_model_name,
        "endpoint_port": record.endpoint_port,
        "base_url": record.base_url,
        "public_ipaddr": record.public_ipaddr,
        "ssh_host": record.ssh_host,
        "ssh_port": record.ssh_port,
        "direct_ssh_host": record.direct_ssh_host,
        "direct_ssh_port": record.direct_ssh_port,
        "ssh_command": ssh_command,
        "last_error": record.last_error,
        "known_base_urls": sorted(_known_vast_base_urls(record)),
        "updated_at": record.updated_at.isoformat(),
    }


def _upsert_vast_router_connection(
    *,
    existing: list[LLMRouterConnectionConfig],
    record: VastInstanceRecord,
) -> list[LLMRouterConnectionConfig]:
    if not record.base_url:
        return [item.model_copy(deep=True) for item in existing]
    filtered = [
        item.model_copy(deep=True)
        for item in existing
        if not (
            item.source_kind == "manual"
            and (
                item.connection_id == "vast-managed"
                or item.base_url == record.base_url
            )
        )
    ]
    filtered.append(
        LLMRouterConnectionConfig(
            connection_id="vast-managed",
            base_url=record.base_url,
            model=record.served_model_name,
            api_key=record.api_key,
            starred=False,
            priority=DEFAULT_VAST_ENDPOINT_PRIORITY,
            source_kind="manual",
            max_concurrent_requests=DEFAULT_VAST_MAX_CONCURRENT,
            supports_image_inputs=True,
        )
    )
    return filtered


def _remove_vast_router_connection(existing: list[LLMRouterConnectionConfig]) -> list[LLMRouterConnectionConfig]:
    return [
        item.model_copy(deep=True)
        for item in existing
        if not (item.source_kind == "manual" and item.connection_id == "vast-managed")
    ]


def _upsert_vast_endpoint_record(
    endpoint_records: list[dict[str, Any]],
    *,
    record: VastInstanceRecord,
    base_url: str,
    api_key: str | None,
    active: bool,
    last_error: str | None,
) -> list[dict[str, Any]]:
    known_base_urls = _known_vast_base_urls(record)
    filtered = [
        item
        for item in endpoint_records
        if item.get("base_url") not in known_base_urls or item.get("base_url") == base_url
    ]
    filtered = [item for item in filtered if item.get("base_url") != base_url]
    filtered.append(
        {
            "base_url": base_url,
            "api_key": api_key,
            "starred": False,
            "priority": DEFAULT_VAST_ENDPOINT_PRIORITY,
            "active": active,
            "manually_disabled": False,
            "last_error": _coerce_optional(last_error),
        }
    )
    return filtered


def _remove_vast_endpoint_record(
    endpoint_records: list[dict[str, Any]],
    *,
    record: VastInstanceRecord | None = None,
    base_url: str | None,
) -> list[dict[str, Any]]:
    known_base_urls = _known_vast_base_urls(record)
    if base_url:
        known_base_urls.add(base_url)
    if not known_base_urls:
        return list(endpoint_records)
    return [item for item in endpoint_records if item.get("base_url") not in known_base_urls]


def _reconcile_vast_instance_state(*, layout: ArtifactLayout) -> VastInstanceRecord | None:
    record = load_vast_instance(layout)
    if record is None or record.instance_id is None:
        return record
    try:
        payload = _parse_vast_json(_run_vast_command("show", "instance", str(record.instance_id), "--raw"))
    except Exception as exc:  # noqa: BLE001
        next_status = VastInstanceStatus.failed
        detail = str(exc).strip()
        lowered = detail.lower()
        if "not found" in lowered or "404" in lowered:
            next_status = VastInstanceStatus.missing
        record = record.model_copy(
            update={
                "status": next_status,
                "last_error": detail or record.last_error,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        save_vast_instance(layout, record)
        return record
    if not isinstance(payload, dict):
        return record
    actual_status = _coerce_optional(payload.get("actual_status"))
    cur_state = _coerce_optional(payload.get("cur_state")) or ""
    next_status = record.status
    last_error = None
    if cur_state.lower() == "stopped":
        next_status = VastInstanceStatus.stopped
    elif actual_status and actual_status.lower() == "running":
        next_status = VastInstanceStatus.running
    elif cur_state.lower() == "running":
        next_status = VastInstanceStatus.starting
    status_msg = _coerce_optional(payload.get("status_msg"))
    if status_msg and next_status != VastInstanceStatus.running:
        last_error = status_msg
    public_ipaddr = _coerce_optional(payload.get("public_ipaddr"))
    direct_ssh_port = payload.get("machine_dir_ssh_port")
    ssh_port = payload.get("ssh_port")
    published_endpoint_port = _vast_published_endpoint_port(payload, record.endpoint_port)
    base_url = f"http://{public_ipaddr}:{published_endpoint_port}" if public_ipaddr else None
    if base_url and next_status == VastInstanceStatus.running:
        probe_error = _probe_vast_endpoint(base_url=base_url, api_key=record.api_key)
        if probe_error is None:
            last_error = None
        else:
            last_error = probe_error
            age_seconds = max(0.0, (datetime.now(timezone.utc) - record.created_at).total_seconds())
            next_status = (
                VastInstanceStatus.failed
                if age_seconds >= DEFAULT_VAST_STARTUP_GRACE_SECONDS
                else VastInstanceStatus.starting
            )
    record = record.model_copy(
        update={
            "base_url": base_url,
            "public_ipaddr": public_ipaddr,
            "ssh_host": _coerce_optional(payload.get("ssh_host")),
            "ssh_port": int(ssh_port) if isinstance(ssh_port, int | float) and ssh_port else record.ssh_port,
            "direct_ssh_host": public_ipaddr,
            "direct_ssh_port": int(direct_ssh_port) if isinstance(direct_ssh_port, int | float) and direct_ssh_port else record.direct_ssh_port,
            "status": next_status,
            "last_error": last_error,
            "updated_at": datetime.now(timezone.utc),
            "metadata": {
                **record.metadata,
                "actual_status": actual_status,
                "cur_state": cur_state,
                "published_endpoint_port": published_endpoint_port,
                "status_msg": status_msg,
                "endpoint_probe_error": last_error,
            },
        }
    )
    save_vast_instance(layout, record)
    return record


def _redirect_with_error(next_url: str, message: str) -> RedirectResponse:
    parsed = urlparse(next_url or "/")
    query = parse_qs(parsed.query, keep_blank_values=True)
    query["router_error"] = [message]
    encoded = urlencode(query, doseq=True)
    location = parsed.path or "/"
    if encoded:
        location = f"{location}?{encoded}"
    return RedirectResponse(location, status_code=303)


def _router_connection_id(
    *,
    explicit_value: str | None,
    base_url: str,
    model: str,
    ordinal: int,
) -> str:
    if explicit_value and explicit_value.strip():
        return explicit_value.strip()
    host_slug = re.sub(r"[^a-z0-9]+", "-", base_url.lower()).strip("-")
    model_slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    candidate = "-".join(part for part in (host_slug, model_slug) if part)
    if not candidate:
        candidate = f"connection-{ordinal}"
    if len(candidate) <= 80:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
    prefix = candidate[: max(1, 80 - len(digest) - 1)].rstrip("-")
    return f"{prefix}-{digest}"


def _router_base_url_id(base_url: str) -> str:
    host_slug = re.sub(r"[^a-z0-9]+", "-", base_url.lower()).strip("-")
    return host_slug[:80] or "router-url"


def _serialize_router_connections(
    snapshot: LLMRouterSnapshot,
) -> list[LLMRouterConnectionConfig]:
    return [
        LLMRouterConnectionConfig(
            connection_id=item.connection_id,
            base_url=item.base_url,
            model=item.model,
            source_kind=item.source_kind,
            worker_id=item.worker_id,
            api_key=item.api_key,
            starred=item.starred,
            priority=item.priority,
            max_concurrent_requests=item.max_concurrent_requests,
            supports_image_inputs=item.supports_image_inputs,
        )
        for item in snapshot.connections
    ]


def _serialize_manual_router_connections(
    snapshot: LLMRouterSnapshot,
) -> list[LLMRouterConnectionConfig]:
    return [
        item for item in _serialize_router_connections(snapshot) if item.source_kind == "manual"
    ]


def _llm_router_endpoint_store_path(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "llm_router_endpoints.json"


def _llm_router_public_access_store_path(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "llm_router_public_access.json"


def _router_error_log_path(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "llm_router_errors.jsonl"


def _router_metrics_history_path(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "llm_router_metrics_history.json"


def _router_request_index_path(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "llm_router_requests.json"


def _router_request_detail_dir(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "llm_router_requests"


def load_llm_router_public_access_settings(layout: ArtifactLayout) -> dict[str, Any]:
    path = _llm_router_public_access_store_path(layout)
    if not path.is_file():
        return {
            "enabled_requested": False,
            "public_api_key": None,
            "password_sha256": None,
            "approval_url": None,
            "last_error": None,
            "updated_at": None,
        }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    return {
        "enabled_requested": bool(raw.get("enabled_requested", False)),
        "public_api_key": _coerce_optional(raw.get("public_api_key")),
        "password_sha256": _coerce_optional(raw.get("password_sha256")),
        "approval_url": _coerce_optional(raw.get("approval_url")),
        "last_error": _coerce_optional(raw.get("last_error")),
        "updated_at": _coerce_optional(raw.get("updated_at")),
    }


def save_llm_router_public_access_settings(layout: ArtifactLayout, settings: dict[str, Any]) -> Path:
    path = _llm_router_public_access_store_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        "enabled_requested": bool(settings.get("enabled_requested", False)),
        "public_api_key": _coerce_optional(settings.get("public_api_key")),
        "password_sha256": _coerce_optional(settings.get("password_sha256")),
        "approval_url": _coerce_optional(settings.get("approval_url")),
        "last_error": _coerce_optional(settings.get("last_error")),
        "updated_at": _coerce_optional(settings.get("updated_at")) or datetime.utcnow().isoformat() + "Z",
    }
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _router_public_access_password_hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _extract_url_from_text(text: str) -> str | None:
    match = re.search(r"https://[^\s]+", text)
    if match is None:
        return None
    return match.group(0).rstrip(".,)")


def _run_tailscale_command(*args: str, timeout_seconds: float = 30.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["tailscale", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return subprocess.CompletedProcess(
            ["tailscale", *args],
            returncode=124,
            stdout=stdout if isinstance(stdout, str) else stdout.decode("utf-8", errors="replace"),
            stderr=stderr if isinstance(stderr, str) else stderr.decode("utf-8", errors="replace"),
        )


def _tailscale_self_dns_name() -> str | None:
    result = _run_tailscale_command("status", "--json")
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except Exception:
        return None
    dns_name = _coerce_optional(payload.get("Self", {}).get("DNSName"))
    if dns_name is None:
        return None
    return dns_name.rstrip(".")


def _router_public_base_url(dns_name: str | None) -> str | None:
    if not dns_name:
        return None
    return f"https://{dns_name}:{ROUTER_PUBLIC_FUNNEL_PORT}{ROUTER_PUBLIC_API_PATH}"


def _tailscale_funnel_status_text() -> str:
    result = _run_tailscale_command("funnel", "status")
    return "\n".join(part for part in (result.stdout, result.stderr) if part).strip()


def _tailscale_funnel_status_payload() -> dict[str, Any]:
    result = _run_tailscale_command("funnel", "status", "--json")
    if result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout or "{}")
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _router_public_funnel_is_active() -> bool:
    payload = _tailscale_funnel_status_payload()
    web = payload.get("Web", {})
    allow_funnel = payload.get("AllowFunnel", {})
    if not isinstance(web, dict) or not isinstance(allow_funnel, dict):
        return False
    for host_port, config in web.items():
        if not str(host_port).endswith(f":{ROUTER_PUBLIC_FUNNEL_PORT}"):
            continue
        if not allow_funnel.get(host_port):
            continue
        handlers = config.get("Handlers", {})
        if not isinstance(handlers, dict):
            continue
        if ROUTER_PUBLIC_FUNNEL_PATH in handlers:
            return True
    return False


def _enable_router_public_funnel(local_target: str) -> tuple[bool, str | None, str | None]:
    result = _run_tailscale_command(
        "funnel",
        "--bg",
        "--yes",
        f"--https={ROUTER_PUBLIC_FUNNEL_PORT}",
        f"--set-path={ROUTER_PUBLIC_FUNNEL_PATH}",
        local_target,
        timeout_seconds=5.0,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode == 0:
        return True, None, None
    approval_url = _extract_url_from_text(output)
    if approval_url and "login.tailscale.com/f/funnel" in approval_url:
        return False, "Approve Funnel in Tailscale, then save again.", approval_url
    return False, output or "Failed to enable Tailscale Funnel.", None


def _disable_router_public_funnel() -> tuple[bool, str | None]:
    result = _run_tailscale_command(
        "funnel",
        f"--https={ROUTER_PUBLIC_FUNNEL_PORT}",
        f"--set-path={ROUTER_PUBLIC_FUNNEL_PATH}",
        "off",
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode == 0:
        return True, None
    if any(marker in output.lower() for marker in ("no serve config", "not serving anything")):
        return True, None
    return False, output or "Failed to disable Tailscale Funnel."


def _router_public_access_payload(layout: ArtifactLayout) -> dict[str, Any]:
    settings = load_llm_router_public_access_settings(layout)
    dns_name = _tailscale_self_dns_name()
    public_base_url = _router_public_base_url(dns_name)
    active = _router_public_funnel_is_active()
    if active:
        status_label = "Live"
        status_class = "running"
    elif settings["enabled_requested"] and settings.get("approval_url"):
        status_label = "Pending approval"
        status_class = "failed"
    elif settings["enabled_requested"]:
        status_label = "Not live"
        status_class = "failed"
    else:
        status_label = "Off"
        status_class = "failed"
    return {
        "enabled_requested": settings["enabled_requested"],
        "public_api_key": settings.get("public_api_key"),
        "password_set": bool(settings.get("password_sha256")),
        "approval_url": settings.get("approval_url"),
        "last_error": settings.get("last_error"),
        "active": active,
        "status_label": status_label,
        "status_class": status_class,
        "dns_name": dns_name,
        "public_base_url": public_base_url,
        "models_url": f"{public_base_url}/models" if public_base_url else None,
        "chat_url": f"{public_base_url}/chat/completions" if public_base_url else None,
        "port": ROUTER_PUBLIC_FUNNEL_PORT,
        "path": ROUTER_PUBLIC_API_PATH,
        "auth_scheme": "None",
    }


def _sync_router_public_access_state(layout: ArtifactLayout, local_target: str) -> dict[str, Any]:
    settings = load_llm_router_public_access_settings(layout)
    if not settings.get("enabled_requested"):
        return settings
    if _router_public_funnel_is_active():
        if settings.get("approval_url") or settings.get("last_error"):
            settings["approval_url"] = None
            settings["last_error"] = None
            save_llm_router_public_access_settings(layout, settings)
        return settings
    ok, error_text, approval_url = _enable_router_public_funnel(local_target)
    settings["approval_url"] = approval_url
    settings["last_error"] = error_text
    if ok:
        settings["approval_url"] = None
        settings["last_error"] = None
    save_llm_router_public_access_settings(layout, settings)
    return settings


def _load_router_errors(layout: ArtifactLayout) -> list[dict[str, Any]]:
    path = _router_error_log_path(layout)
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    try:
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    payload.setdefault("id", idx)
                    entries.append(payload)
            except Exception:
                continue
    except Exception:
        return []
    return entries


def _router_error_count(layout: ArtifactLayout) -> int:
    path = _router_error_log_path(layout)
    if not path.is_file():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def _clear_router_errors(layout: ArtifactLayout) -> None:
    path = _router_error_log_path(layout)
    try:
        if path.is_file():
            path.unlink()
    except Exception:
        return


def _log_router_request_error(
    *,
    layout: ArtifactLayout,
    payload: dict[str, Any],
    error: Exception,
) -> None:
    try:
        path = _router_error_log_path(layout)
        path.parent.mkdir(parents=True, exist_ok=True)
        upstream_request_seconds = router_metadata.get("request_duration_seconds")
        if not isinstance(upstream_request_seconds, (int, float)):
            upstream_request_seconds = None
        end_to_end_request_seconds = max((completed_at - started_at).total_seconds(), 0.0)
        queue_wait_seconds = None
        if upstream_request_seconds is not None:
            queue_wait_seconds = max(end_to_end_request_seconds - float(upstream_request_seconds), 0.0)
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": str(error),
            "payload": payload,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # best-effort logging; swallow any file errors
        return


def load_llm_router_endpoint_records(layout: ArtifactLayout) -> list[dict[str, Any]]:
    path = _llm_router_endpoint_store_path(layout)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Malformed dashboard LLM router endpoint store: {path}")
    records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        base_url = str(item.get("base_url") or "").strip()
        if not base_url:
            continue
        records.append(
            {
                "base_url": base_url,
                "api_key": _coerce_optional(item.get("api_key")),
                "starred": bool(item.get("starred", False)),
                "priority": max(1, min(99, int(item.get("priority", 2) or 2))),
                "active": bool(item.get("active", True)),
                "manually_disabled": bool(item.get("manually_disabled", False)),
                "last_error": _coerce_optional(item.get("last_error")),
            }
        )
    return records


def save_llm_router_endpoint_records(
    layout: ArtifactLayout,
    records: list[dict[str, Any]],
) -> Path:
    path = _llm_router_endpoint_store_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    seen: set[str] = set()
    for item in records:
        base_url = str(item.get("base_url") or "").strip()
        if not base_url or base_url in seen:
            continue
        seen.add(base_url)
        normalized.append(
            {
                "base_url": base_url,
                "api_key": _coerce_optional(item.get("api_key")),
                "starred": bool(item.get("starred", False)),
                "priority": max(1, min(99, int(item.get("priority", 2) or 2))),
                "active": bool(item.get("active", True)),
                "manually_disabled": bool(item.get("manually_disabled", False)),
                "last_error": _coerce_optional(item.get("last_error")),
            }
        )
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _router_models_probe_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if urlparse(trimmed).path.rstrip("/").endswith("/v1"):
        return f"{trimmed}/models"
    return f"{trimmed}/v1/models"


async def _probe_router_endpoint_health(base_url: str, api_key: str | None) -> tuple[bool, str | None]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=5.0, headers=headers) as client:
        try:
            response = await client.get(_router_models_probe_url(base_url))
            response.raise_for_status()
            payload = response.json()
            if not (
                isinstance(payload, list)
                or (isinstance(payload, dict) and isinstance(payload.get("data"), list))
            ):
                return False, "Health probe did not return a models list."
            return True, None
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            return False, detail or str(exc)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)


def _group_router_connections(
    connections: list[LLMRouterConnectionSnapshot],
    *,
    endpoint_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for connection in connections:
        endpoint = grouped.setdefault(
            connection.base_url,
            {
                "endpoint_id": _router_base_url_id(connection.base_url),
                "base_url": connection.base_url,
                "source_kind": connection.source_kind,
                "worker_ids": [],
                "api_key_present": False,
                "starred": False,
                "priority": int(connection.priority or 2),
                "in_flight_requests": 0,
                "max_concurrent_requests": 0,
                "request_count": 0,
                "total_completion_tokens": 0,
                "active_models": 0,
                "disabled_models": 0,
                "inactive_models": 0,
                "last_error_messages": [],
                "models": [],
            },
        )
        if connection.worker_id:
            endpoint["worker_ids"].append(connection.worker_id)
        endpoint["api_key_present"] = endpoint["api_key_present"] or bool(connection.api_key)
        endpoint["starred"] = endpoint["starred"] or bool(connection.starred)
        endpoint["priority"] = min(int(endpoint.get("priority", 2)), int(connection.priority or 2))
        endpoint["in_flight_requests"] += int(connection.in_flight_requests)
        if connection.active and not connection.manually_disabled:
            endpoint["max_concurrent_requests"] += int(connection.max_concurrent_requests)
        endpoint["request_count"] += int(connection.telemetry.request_count)
        endpoint["total_completion_tokens"] += int(connection.telemetry.total_completion_tokens)
        if connection.manually_disabled:
            endpoint["disabled_models"] += 1
        elif connection.active:
            endpoint["active_models"] += 1
        else:
            endpoint["inactive_models"] += 1
        if connection.last_error:
            endpoint["last_error_messages"].append(connection.last_error)
        endpoint["models"].append(connection)
    endpoints: list[dict[str, Any]] = []
    for endpoint in grouped.values():
        models = sorted(endpoint["models"], key=lambda item: item.model.lower())
        endpoint["models"] = models
        model_count = len(models)
        if endpoint["disabled_models"] == model_count and model_count > 0:
            status_label = "Disabled"
            status_class = "failed"
        elif endpoint["active_models"] > 0:
            status_label = "Active"
            status_class = "running"
        else:
            status_label = "Inactive"
            status_class = "failed"
        endpoint["status_label"] = status_label
        endpoint["status_class"] = status_class
        endpoint["source_label"] = "Worker" if endpoint.get("source_kind") == "worker" else "Manual"
        endpoint["model_count"] = model_count
        endpoint["completion_tokens_label"] = _format_compact_metric(
            int(endpoint["total_completion_tokens"])
        )
        endpoint["last_error"] = (
            endpoint["last_error_messages"][0] if endpoint["last_error_messages"] else None
        )
        endpoint["default_open"] = False
        endpoints.append(endpoint)
    for record in endpoint_records or []:
        base_url = record["base_url"]
        if base_url in grouped:
            endpoint = grouped[base_url]
            endpoint["api_key_present"] = endpoint["api_key_present"] or bool(record.get("api_key"))
            endpoint["starred"] = endpoint["starred"] or bool(record.get("starred"))
            endpoint["priority"] = min(
                int(endpoint.get("priority", 2)),
                max(1, min(99, int(record.get("priority", 2) or 2))),
            )
            if endpoint["last_error"] is None:
                endpoint["last_error"] = record.get("last_error")
            continue
        endpoints.append(
            {
                "endpoint_id": _router_base_url_id(base_url),
                "base_url": base_url,
                "source_kind": "manual",
                "worker_ids": [],
                "api_key_present": bool(record.get("api_key")),
                "starred": bool(record.get("starred")),
                "priority": max(1, min(99, int(record.get("priority", 2) or 2))),
                "in_flight_requests": 0,
                "max_concurrent_requests": 0,
                "request_count": 0,
                "total_completion_tokens": 0,
                "active_models": 0,
                "disabled_models": 0,
                "inactive_models": 0,
                "last_error_messages": [],
                "models": [],
                "status_label": (
                    "Disabled"
                    if record.get("manually_disabled")
                    else ("Active" if record.get("active", True) else "Inactive")
                ),
                "status_class": (
                    "failed"
                    if record.get("manually_disabled") or not record.get("active", True)
                    else "running"
                ),
                "source_label": "Manual",
                "model_count": 0,
                "completion_tokens_label": _format_compact_metric(0),
                "last_error": record.get("last_error"),
                "default_open": False,
            }
        )
    return sorted(endpoints, key=lambda item: (int(item.get("priority", 2)), item["base_url"].lower()))


def _format_compact_metric(value: int) -> str:
    abs_value = abs(int(value))
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _upsert_router_connection(
    existing: list[LLMRouterConnectionConfig],
    new_connection: LLMRouterConnectionConfig,
) -> list[LLMRouterConnectionConfig]:
    merged = [item for item in existing if item.connection_id != new_connection.connection_id]
    inherited_api_key = new_connection.api_key
    inherited_starred = new_connection.starred
    inherited_priority = new_connection.priority
    if inherited_api_key is None:
        for item in existing:
            if item.base_url == new_connection.base_url and item.api_key:
                inherited_api_key = item.api_key
                break
    if not inherited_starred:
        for item in existing:
            if item.base_url == new_connection.base_url and item.starred:
                inherited_starred = True
                break
    if inherited_priority == new_connection.priority:
        for item in existing:
            if item.base_url == new_connection.base_url:
                inherited_priority = item.priority
                break
    if (
        inherited_api_key != new_connection.api_key
        or inherited_starred != new_connection.starred
        or inherited_priority != new_connection.priority
    ):
        new_connection = new_connection.model_copy(
            update={
                "api_key": inherited_api_key,
                "starred": inherited_starred,
                "priority": inherited_priority,
            }
        )
    merged.append(new_connection)
    return merged


def _stored_router_api_key(
    *,
    base_url: str,
    connections: list[LLMRouterConnectionConfig] | None = None,
    endpoint_records: list[dict[str, Any]] | None = None,
) -> str | None:
    for item in connections or []:
        if item.base_url == base_url and item.api_key:
            return item.api_key
    for item in endpoint_records or []:
        if item.get("base_url") == base_url and item.get("api_key"):
            return _coerce_optional(item.get("api_key"))
    return None


def _stored_router_starred(
    *,
    base_url: str,
    connections: list[LLMRouterConnectionConfig] | None = None,
    endpoint_records: list[dict[str, Any]] | None = None,
) -> bool:
    for item in connections or []:
        if item.base_url == base_url and item.starred:
            return True
    for item in endpoint_records or []:
        if item.get("base_url") == base_url and item.get("starred"):
            return True
    return False


def _stored_router_priority(
    *,
    base_url: str,
    connections: list[LLMRouterConnectionConfig] | None = None,
    endpoint_records: list[dict[str, Any]] | None = None,
) -> int:
    for item in connections or []:
        if item.base_url == base_url:
            return int(item.priority)
    for item in endpoint_records or []:
        if item.get("base_url") == base_url:
            return max(1, min(99, int(item.get("priority", 2) or 2)))
    return 2


def _parse_single_router_connection(
    parsed_form: dict[str, list[str]],
) -> LLMRouterConnectionConfig | None:
    def _first(name: str) -> str:
        values = parsed_form.get(name, [])
        if not values:
            return ""
        return values[0]

    base_url = _first("connection_base_url").strip()
    model = _first("connection_model").strip()
    api_key = _first("connection_api_key").strip()
    max_concurrent = _first("connection_max_concurrent").strip()
    explicit_id = _first("connection_id").strip()
    priority = _first("connection_priority").strip()
    supports_image_inputs = bool(parsed_form.get("connection_supports_images"))
    if not base_url and not model and not max_concurrent:
        return None
    return LLMRouterConnectionConfig(
        connection_id=_router_connection_id(
            explicit_value=explicit_id or None,
            base_url=base_url,
            model=model,
            ordinal=1,
        ),
        base_url=base_url,
        model=model,
        api_key=api_key or None,
        starred=False,
        priority=int(priority or "2"),
        max_concurrent_requests=int(max_concurrent or "1"),
        supports_image_inputs=supports_image_inputs,
    )


def _router_overview_metrics(snapshot: LLMRouterSnapshot) -> dict[str, Any]:
    total_prompt_tokens = sum(
        int(connection.telemetry.total_prompt_tokens)
        for connection in snapshot.connections
    )
    total_completion_tokens = sum(
        int(connection.telemetry.total_completion_tokens)
        for connection in snapshot.connections
    )
    active_connections = [
        connection
        for connection in snapshot.connections
        if connection.active and not connection.manually_disabled
    ]
    total_estimated_throughput = 0.0
    recent_request_durations: list[float] = []
    recent_completion_tokens: list[float] = []
    total_in_flight_requests = 0
    for connection in active_connections:
        total_in_flight_requests += int(connection.in_flight_requests)
        samples = connection.telemetry.recent_request_samples[-25:]
        recent_request_durations.extend(
            float(sample.get("request_seconds") or 0.0)
            for sample in samples
        )
        recent_completion_tokens.extend(
            float(sample.get("completion_tokens") or 0.0)
            for sample in samples
        )
        if samples:
            total_completion = sum(float(sample.get("completion_tokens") or 0.0) for sample in samples)
            total_seconds = sum(max(float(sample.get("request_seconds") or 0.0), 1e-9) for sample in samples)
            average_concurrency = (
                sum(max(int(sample.get("concurrent_requests_seen") or 0), 0) for sample in samples) / len(samples)
            )
            if total_seconds > 0.0 and average_concurrency > 0.0:
                total_estimated_throughput += (total_completion / total_seconds) * average_concurrency
    if recent_request_durations:
        recent_request_durations = recent_request_durations[-25:]
    average_request_seconds = (
        sum(recent_request_durations) / len(recent_request_durations)
        if recent_request_durations
        else 0.0
    )
    total_recent_seconds = sum(recent_request_durations)
    total_recent_completion = sum(recent_completion_tokens[-25:]) if recent_completion_tokens else 0.0
    average_tokens_per_second = (
        total_recent_completion / total_recent_seconds if total_recent_seconds > 0 else 0.0
    )
    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_prompt_tokens_label": _format_compact_metric(total_prompt_tokens),
        "total_completion_tokens": total_completion_tokens,
        "total_completion_tokens_label": _format_compact_metric(total_completion_tokens),
        "average_request_seconds": round(average_request_seconds, 2),
        "average_output_tokens_per_second": round(average_tokens_per_second, 2),
        "total_output_tokens_per_second": round(total_estimated_throughput, 2),
        "total_in_flight_requests": total_in_flight_requests,
    }


def _serialize_router_telemetry(telemetry: Any) -> dict[str, Any]:
    if hasattr(telemetry, "model_dump"):
        payload = telemetry.model_dump(mode="json")
    elif isinstance(telemetry, dict):
        payload = dict(telemetry)
    else:
        payload = {}
    average_recent_request_seconds = getattr(telemetry, "average_recent_request_seconds", None)
    average_output_tokens_per_second = getattr(telemetry, "average_output_tokens_per_second", None)
    average_context_tokens = getattr(telemetry, "average_context_tokens", None)
    average_concurrent_requests = getattr(telemetry, "average_concurrent_requests", None)
    payload["average_recent_request_seconds"] = average_recent_request_seconds
    payload["average_output_tokens_per_second"] = average_output_tokens_per_second
    payload["average_context_tokens"] = average_context_tokens
    payload["average_concurrent_requests"] = average_concurrent_requests
    return payload


def _serialize_router_connection_snapshot(connection: Any) -> dict[str, Any]:
    if hasattr(connection, "model_dump"):
        payload = connection.model_dump(mode="json")
    elif isinstance(connection, dict):
        payload = dict(connection)
    else:
        payload = {}
    payload["telemetry"] = _serialize_router_telemetry(getattr(connection, "telemetry", payload.get("telemetry")))
    return payload


def _serialize_router_panel_payload(panel: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(panel)
    serialized["connections"] = [
        _serialize_router_connection_snapshot(connection)
        for connection in panel.get("connections", [])
    ]
    serialized["worker_connections"] = [
        _serialize_router_connection_snapshot(connection)
        for connection in panel.get("worker_connections", [])
    ]
    serialized["workers"] = [
        worker.model_dump(mode="json") if hasattr(worker, "model_dump") else dict(worker)
        for worker in panel.get("workers", [])
    ]
    serialized_endpoints: list[dict[str, Any]] = []
    for endpoint in panel.get("endpoints", []):
        item = dict(endpoint)
        item["models"] = [
            _serialize_router_connection_snapshot(connection)
            for connection in endpoint.get("models", [])
        ]
        serialized_endpoints.append(item)
    serialized["endpoints"] = serialized_endpoints
    return jsonable_encoder(serialized)


def _router_panel_payload(
    *,
    router_base_url: str,
    shared_model: str,
    snapshot: LLMRouterSnapshot,
    workers: list[LLMWorkerRecord],
    public_access: dict[str, Any],
    endpoint_records: list[dict[str, Any]] | None = None,
    error: str | None = None,
    error_count: int = 0,
) -> dict[str, Any]:
    overview_metrics = _router_overview_metrics(snapshot)
    endpoints = _group_router_connections(snapshot.connections, endpoint_records=endpoint_records)
    endpoints = [
        endpoint
        for endpoint in endpoints
        if endpoint.get("source_kind") == "manual"
    ]
    endpoints = [
        endpoint
        for endpoint in endpoints
        if not (
            endpoint.get("status_label") == "Inactive"
            and endpoint.get("model_count", 0) == 0
            and endpoint.get("in_flight_requests", 0) == 0
        )
    ]
    online_workers = [item for item in workers if item.status.value == "online"]
    worker_connections = [
        connection
        for connection in snapshot.connections
        if connection.source_kind == "worker" and connection.active
    ]
    return {
        "base_url": router_base_url.rstrip("/"),
        "shared_model": shared_model,
        "public_access": public_access,
        "connections": snapshot.connections,
        "endpoints": endpoints,
        "workers": online_workers,
        "worker_connections": worker_connections,
        "worker_counts": {
            "online": len(online_workers),
            "offline": 0,
            "total": len(online_workers),
        },
        "waiting_requests": snapshot.waiting_requests,
        **overview_metrics,
        "rendered_at": datetime.now().strftime("%H:%M:%S"),
        "error": error,
        "error_count": error_count,
        "canonical_models": sorted(
            {
                (conn.canonical_model or conn.model, conn.active)
                for conn in snapshot.connections
            },
            key=lambda item: (0 if item[1] else 1, item[0]),
        ),
    }


def _extract_public_api_key(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token or None
    api_key = request.headers.get("x-api-key", "").strip()
    return api_key or None


def _request_uses_public_router_host(request: Request, layout: ArtifactLayout) -> bool:
    settings = load_llm_router_public_access_settings(layout)
    if not settings.get("enabled_requested"):
        return False
    dns_name = _tailscale_self_dns_name()
    if not dns_name:
        return False
    expected_host = f"{dns_name}:{ROUTER_PUBLIC_FUNNEL_PORT}".lower()
    observed_hosts = {
        request.headers.get("host", "").strip().lower(),
        request.headers.get("x-forwarded-host", "").strip().lower(),
    }
    return expected_host in observed_hosts


def _require_public_api_key(request: Request, layout: ArtifactLayout) -> None:
    settings = load_llm_router_public_access_settings(layout)
    if not settings.get("enabled_requested"):
        raise HTTPException(status_code=404, detail="Public router access is not enabled.")
    return


async def _persist_router_snapshot(*, layout: ArtifactLayout, router: DashboardLLMRouter) -> None:
    save_llm_router_snapshot(layout, router.snapshot())


def _sanitize_router_request_headers(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    hidden = {"authorization", "x-api-key", "cookie"}
    sanitized: dict[str, str] = {}
    items = headers.items() if hasattr(headers, "items") else []
    for name, value in items:
        if str(name).lower() in hidden:
            continue
        sanitized[str(name)] = str(value)
    return sanitized


def create_router_app(*, layout: ArtifactLayout | None = None) -> FastAPI:
    resolved_layout = layout or get_artifact_layout()
    router_metrics_history = RouterMetricsHistoryStore(_router_metrics_history_path(resolved_layout))
    router_request_store = RouterRequestStore(
        index_path=_router_request_index_path(resolved_layout),
        detail_dir=_router_request_detail_dir(resolved_layout),
        max_entries=ROUTER_REQUEST_LOG_MAX_ENTRIES,
    )
    router_snapshot = load_llm_router_snapshot(resolved_layout)
    worker_registry = LLMWorkerRegistry(
        workers=load_llm_workers(resolved_layout),
        heartbeat_timeout_seconds=ROUTER_WORKER_HEARTBEAT_TIMEOUT_SECONDS,
    )
    worker_request_queue = WorkerRequestQueue()
    endpoint_records = load_llm_router_endpoint_records(resolved_layout)
    manual_router_connections = _serialize_manual_router_connections(router_snapshot)
    manual_router_connections[:] = _remove_vast_router_connection(manual_router_connections)
    endpoint_records[:] = _remove_vast_endpoint_record(endpoint_records, base_url=None)
    save_llm_router_endpoint_records(resolved_layout, endpoint_records)

    async def _handle_pull_worker_request(
        config: LLMRouterConnectionConfig,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        worker_id = (config.worker_id or "").strip()
        if not worker_id:
            raise RuntimeError("Pull-request worker connection is missing worker_id")
        request_id, body = await worker_request_queue.enqueue_and_wait(
            worker_id=worker_id,
            payload=payload,
            timeout_seconds=600.0,
        )
        router_metadata = body.get("router")
        if not isinstance(router_metadata, dict):
            router_metadata = {}
        router_metadata["worker_request_id"] = request_id
        body["router"] = router_metadata
        return body

    dashboard_llm_router = DashboardLLMRouter(
        connections=[
            *[
                item.model_copy(
                    update={
                        "starred": _stored_router_starred(
                            base_url=item.base_url,
                            endpoint_records=endpoint_records,
                        )
                        or item.starred,
                        "priority": _stored_router_priority(
                            base_url=item.base_url,
                            endpoint_records=endpoint_records,
                        ),
                    }
                )
                for item in manual_router_connections
            ],
            *worker_registry.active_connection_configs(),
        ],
        initial_snapshot=router_snapshot,
        timeout_seconds=600.0,
        worker_request_handler=_handle_pull_worker_request,
    )
    shared_llm = load_shared_llm_config(resolved_layout.config_root / "LLM.yaml")

    app = FastAPI(title="LLM Router")
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
    app.state.router_health_task = None
    app.state.router_no_active_since = None
    app.state.router_public_access_last_sync = 0.0
    app.state.router_panel_cache = None

    def _combined_router_connections() -> list[LLMRouterConnectionConfig]:
        return [
            *[item.model_copy(deep=True) for item in manual_router_connections],
            *worker_registry.active_connection_configs(),
        ]

    async def _persist_worker_registry() -> None:
        save_llm_workers(resolved_layout, worker_registry.list_workers())

    async def _sync_router_sources(*, refresh_health: bool = False) -> None:
        await dashboard_llm_router.replace_connections(_combined_router_connections())
        if refresh_health:
            await dashboard_llm_router.refresh_connection_health()
        await _persist_router_snapshot(layout=resolved_layout, router=dashboard_llm_router)

    def _update_router_connection_outage_state() -> None:
        snapshot = dashboard_llm_router.snapshot()
        if any(item.active for item in snapshot.connections):
            app.state.router_no_active_since = None
            return
        if app.state.router_no_active_since is None:
            app.state.router_no_active_since = time.monotonic()

    def _router_panel_data(*, error: str | None = None) -> dict[str, Any]:
        cache_entry = app.state.router_panel_cache
        now = time.monotonic()
        if error is None and cache_entry is not None and (now - float(cache_entry["captured_at"])) < 2.0:
            return cache_entry["payload"]
        workers = worker_registry.list_workers()
        modal_launches = _reconcile_modal_worker_launches(
            layout=resolved_layout,
            workers=workers,
        )
        payload = _router_panel_payload(
            router_base_url=str(shared_llm.base_url).rstrip("/"),
            shared_model=shared_llm.model,
            snapshot=dashboard_llm_router.snapshot(),
            workers=workers,
            public_access=_router_public_access_payload(resolved_layout),
            endpoint_records=load_llm_router_endpoint_records(resolved_layout),
            error=error,
            error_count=_router_error_count(resolved_layout),
        )
        if error is None:
            app.state.router_panel_cache = {
                "captured_at": now,
                "payload": payload,
            }
        return payload

    def _router_request_path_is_public(request: Request) -> bool:
        return request.url.path.startswith(ROUTER_PUBLIC_API_PATH)

    def _record_router_request(
        *,
        request: Request,
        payload: dict[str, Any],
        started_at: datetime,
        completed_at: datetime,
        status: str,
        http_status_code: int,
        response_body: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        router_metadata = response_body.get("router") if isinstance(response_body, dict) else None
        if not isinstance(router_metadata, dict):
            router_metadata = {}
        usage = response_body.get("usage") if isinstance(response_body, dict) else None
        if not isinstance(usage, dict):
            usage = {}
        upstream_request_seconds = router_metadata.get("request_duration_seconds")
        if not isinstance(upstream_request_seconds, (int, float)):
            upstream_request_seconds = None
        end_to_end_request_seconds = max((completed_at - started_at).total_seconds(), 0.0)
        queue_wait_seconds = None
        if upstream_request_seconds is not None:
            queue_wait_seconds = max(
                end_to_end_request_seconds - float(upstream_request_seconds),
                0.0,
            )
        entry = {
            "id": uuid4().hex,
            "status": status,
            "http_status_code": int(http_status_code),
            "started_at": started_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "completed_at": completed_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "request_path": request.url.path,
            "query_params": dict(request.query_params),
            "received_via_public_router": _router_request_path_is_public(request),
            "request_headers": _sanitize_router_request_headers(request.headers),
            "request_client": {
                "host": request.client.host if request.client is not None else None,
                "port": request.client.port if request.client is not None else None,
            },
            "requested_model": payload.get("model"),
            "request": payload,
            "response": response_body if isinstance(response_body, dict) else None,
            "error": error_message,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "connection": {
                "connection_id": router_metadata.get("connection_id"),
                "base_url": router_metadata.get("base_url"),
                "selected_model": router_metadata.get("selected_model"),
                "source_kind": router_metadata.get("source_kind"),
                "transport_mode": router_metadata.get("transport_mode"),
                "worker_id": router_metadata.get("worker_id"),
                "worker_request_id": router_metadata.get("worker_request_id"),
            },
            "timing": {
                "router_request_seconds": round(
                    float(upstream_request_seconds)
                    if upstream_request_seconds is not None
                    else end_to_end_request_seconds,
                    6,
                ),
                "upstream_request_seconds": round(float(upstream_request_seconds), 6)
                if upstream_request_seconds is not None
                else None,
                "end_to_end_request_seconds": round(end_to_end_request_seconds, 6),
                "queue_wait_seconds": round(queue_wait_seconds, 6)
                if queue_wait_seconds is not None
                else None,
            },
            "response_headers": (
                {
                    "X-LLM-Connection-Id": router_metadata.get("connection_id"),
                    "X-LLM-Server": router_metadata.get("base_url"),
                    "X-LLM-Model": router_metadata.get("selected_model"),
                }
                if status == "completed"
                else {}
            ),
        }
        try:
            router_request_store.record(entry)
        except Exception:
            return

    async def _router_health_loop() -> None:
        while True:
            try:
                stale_workers = worker_registry.prune_stale_workers()
                if stale_workers:
                    await _persist_worker_registry()
                    await _sync_router_sources(refresh_health=False)
                disconnected_modal_workers = False
                modal_app_states = _modal_app_state_map(resolved_layout)
                for worker in worker_registry.list_workers():
                    if worker.status.value != "online":
                        continue
                    metadata = worker.metadata if isinstance(worker.metadata, dict) else {}
                    if metadata.get("launcher") != "modal_worker":
                        continue
                    modal_app_id = _coerce_optional(metadata.get("modal_app_id"))
                    if _modal_app_state_is_live(modal_app_states.get(modal_app_id)):
                        continue
                    try:
                        worker_registry.disconnect(
                            worker.worker_id,
                            reason="Modal app is no longer live.",
                        )
                    except KeyError:
                        continue
                    disconnected_modal_workers = True
                if disconnected_modal_workers:
                    await _persist_worker_registry()
                    await _sync_router_sources(refresh_health=False)
                await dashboard_llm_router.refresh_connection_health()
                await _persist_router_snapshot(layout=resolved_layout, router=dashboard_llm_router)
                router_metrics = _router_overview_metrics(dashboard_llm_router.snapshot())
                await asyncio.to_thread(
                    router_metrics_history.record,
                    in_flight_requests=router_metrics["total_in_flight_requests"],
                    throughput=router_metrics["total_output_tokens_per_second"],
                )
                app.state.router_panel_cache = None
                _update_router_connection_outage_state()
                now = time.monotonic()
                if now - float(app.state.router_public_access_last_sync or 0.0) >= 15.0:
                    await asyncio.to_thread(
                        _sync_router_public_access_state,
                        resolved_layout,
                        "http://127.0.0.1:8788",
                    )
                    app.state.router_public_access_last_sync = now
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(ROUTER_HEALTHCHECK_INTERVAL_SECONDS)

    @app.on_event("startup")
    async def router_startup() -> None:
        await _sync_router_sources(refresh_health=False)
        await dashboard_llm_router.refresh_connection_health()
        await _persist_router_snapshot(layout=resolved_layout, router=dashboard_llm_router)
        await _persist_worker_registry()
        _reconcile_modal_worker_launches(
            layout=resolved_layout,
            workers=worker_registry.list_workers(),
        )
        router_metrics = _router_overview_metrics(dashboard_llm_router.snapshot())
        await asyncio.to_thread(
            router_metrics_history.record,
            in_flight_requests=router_metrics["total_in_flight_requests"],
            throughput=router_metrics["total_output_tokens_per_second"],
        )
        _update_router_connection_outage_state()
        await asyncio.to_thread(
            _sync_router_public_access_state,
            resolved_layout,
            "http://127.0.0.1:8788",
        )
        app.state.router_public_access_last_sync = time.monotonic()
        app.state.router_health_task = asyncio.create_task(_router_health_loop())

    @app.on_event("shutdown")
    async def router_shutdown() -> None:
        task = app.state.router_health_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await _persist_router_snapshot(layout=resolved_layout, router=dashboard_llm_router)
        await _persist_worker_registry()
        await dashboard_llm_router.aclose()

    @app.post("/router/public-access")
    async def router_public_access_update(request: Request) -> RedirectResponse:
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        mode = (parsed_form.get("mode", [""])[0] or "").strip().lower()
        enable_requested = bool(parsed_form.get("public_enabled"))
        raw_password = (parsed_form.get("public_api_key", [""])[0] or "").strip()
        current = load_llm_router_public_access_settings(resolved_layout)
        next_settings = dict(current)
        if raw_password:
            next_settings["public_api_key"] = raw_password
            next_settings["password_sha256"] = _router_public_access_password_hash(raw_password)
        if mode == "save":
            save_llm_router_public_access_settings(resolved_layout, next_settings)
            return RedirectResponse(next_url, status_code=303)
        if enable_requested:
            next_settings["enabled_requested"] = True
            local_target = f"http://127.0.0.1:{request.url.port}"
            ok, error_text, approval_url = await asyncio.to_thread(_enable_router_public_funnel, local_target)
            next_settings["approval_url"] = approval_url
            next_settings["last_error"] = error_text
            save_llm_router_public_access_settings(resolved_layout, next_settings)
            if not ok and approval_url is None:
                return _redirect_with_error(next_url, error_text or "Failed to enable public router access.")
            return RedirectResponse(next_url, status_code=303)
        next_settings["enabled_requested"] = False
        next_settings["approval_url"] = None
        next_settings["last_error"] = None
        ok, error_text = await asyncio.to_thread(_disable_router_public_funnel)
        save_llm_router_public_access_settings(resolved_layout, next_settings)
        if not ok:
            return _redirect_with_error(next_url, error_text or "Failed to disable public router access.")
        return RedirectResponse(next_url, status_code=303)

    @app.post("/router/errors/clear")
    async def router_errors_clear() -> RedirectResponse:
        _clear_router_errors(resolved_layout)
        return RedirectResponse("/router/errors", status_code=303)

    @app.get("/router/snapshot")
    async def router_snapshot_view() -> JSONResponse:
        return JSONResponse(dashboard_llm_router.snapshot().model_dump(mode="json"))

    @app.get("/api/router/panel")
    async def router_panel_view() -> JSONResponse:
        return JSONResponse(_serialize_router_panel_payload(_router_panel_data()))

    @app.get("/api/router/history")
    async def router_history_view() -> JSONResponse:
        return JSONResponse(router_metrics_history.payload())

    @app.get("/api/router/requests")
    async def router_requests_view(limit: int | None = None) -> JSONResponse:
        return JSONResponse(router_request_store.payload(limit=limit))

    @app.get("/api/router/requests/{request_id}")
    async def router_request_entry_view(request_id: str) -> JSONResponse:
        entry = router_request_store.detail(request_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Request entry not found")
        return JSONResponse(entry)

    @app.get("/api/router/errors")
    async def router_errors_view() -> JSONResponse:
        return JSONResponse({"errors": list(reversed(_load_router_errors(resolved_layout)))})

    @app.get("/api/router/errors/{entry_id}")
    async def router_error_entry_view(entry_id: int) -> JSONResponse:
        entries = _load_router_errors(resolved_layout)
        entry = next((item for item in entries if item.get("id") == entry_id), None)
        if entry is None:
            raise HTTPException(status_code=404, detail="Error entry not found")
        return JSONResponse(entry)

    @app.post("/router/connections")
    async def add_router_connection(request: Request) -> Response:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
            try:
                new_connection = LLMRouterConnectionConfig.model_validate(payload)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            new_connection = new_connection.model_copy(update={"source_kind": "manual", "worker_id": None})
            snapshot = dashboard_llm_router.snapshot()
            existing = _serialize_manual_router_connections(snapshot)
            endpoint_records = load_llm_router_endpoint_records(resolved_layout)
            preserved_api_key = new_connection.api_key or _stored_router_api_key(
                base_url=new_connection.base_url,
                connections=existing,
                endpoint_records=endpoint_records,
            )
            if preserved_api_key != new_connection.api_key:
                new_connection = new_connection.model_copy(update={"api_key": preserved_api_key})
            manual_router_connections[:] = _upsert_router_connection(existing, new_connection)
            await _sync_router_sources(refresh_health=True)
            endpoint_records = [item for item in endpoint_records if item["base_url"] != new_connection.base_url]
            active, last_error = await _probe_router_endpoint_health(new_connection.base_url, new_connection.api_key)
            endpoint_records.append(
                {
                    "base_url": new_connection.base_url,
                    "api_key": new_connection.api_key,
                    "starred": _stored_router_starred(
                        base_url=new_connection.base_url,
                        connections=existing,
                        endpoint_records=endpoint_records,
                    )
                    or new_connection.starred,
                    "priority": new_connection.priority,
                    "active": active,
                    "manually_disabled": False,
                    "last_error": last_error,
                }
            )
            save_llm_router_endpoint_records(resolved_layout, endpoint_records)
            return JSONResponse(dashboard_llm_router.snapshot().model_dump(mode="json"))

        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        base_url = (parsed_form.get("connection_base_url", [""])[0] or "").strip()
        api_key = (parsed_form.get("connection_api_key", [""])[0] or "").strip() or None
        model = (parsed_form.get("connection_model", [""])[0] or "").strip()
        try:
            max_concurrent = int((parsed_form.get("connection_max_concurrent", ["1"])[0] or "1").strip() or "1")
        except ValueError:
            return _redirect_with_error(next_url, "Max concurrent must be an integer.")
        try:
            priority = int((parsed_form.get("connection_priority", ["2"])[0] or "2").strip() or "2")
            priority = max(1, min(99, priority))
        except ValueError:
            return _redirect_with_error(next_url, "Priority must be an integer.")
        if not base_url:
            return _redirect_with_error(next_url, "Base URL must be non-empty.")
        endpoint_records = load_llm_router_endpoint_records(resolved_layout)
        if not model:
            active, last_error = await _probe_router_endpoint_health(base_url, api_key)
            endpoint_records = [item for item in endpoint_records if item["base_url"] != base_url]
            endpoint_records.append(
                {
                    "base_url": base_url,
                    "api_key": api_key,
                    "starred": _stored_router_starred(
                        base_url=base_url,
                        endpoint_records=endpoint_records,
                    ),
                    "priority": priority,
                    "active": active,
                    "manually_disabled": False,
                    "last_error": last_error,
                }
            )
            save_llm_router_endpoint_records(resolved_layout, endpoint_records)
            return RedirectResponse(next_url, status_code=303)
        try:
            existing = _serialize_manual_router_connections(dashboard_llm_router.snapshot())
            api_key = api_key or _stored_router_api_key(
                base_url=base_url,
                connections=existing,
                endpoint_records=endpoint_records,
            )
            new_connection = LLMRouterConnectionConfig(
                connection_id=_router_connection_id(explicit_value=None, base_url=base_url, model=model, ordinal=1),
                base_url=base_url,
                model=model,
                api_key=api_key,
                starred=_stored_router_starred(
                    base_url=base_url,
                    connections=existing,
                    endpoint_records=endpoint_records,
                ),
                priority=priority,
                source_kind="manual",
                max_concurrent_requests=max_concurrent,
                supports_image_inputs=bool(parsed_form.get("connection_supports_images")),
            )
            manual_router_connections[:] = _upsert_router_connection(existing, new_connection)
            await _sync_router_sources(refresh_health=True)
            endpoint_records = [item for item in endpoint_records if item["base_url"] != base_url]
            active, last_error = await _probe_router_endpoint_health(base_url, api_key)
            endpoint_records.append(
                {
                    "base_url": base_url,
                    "api_key": api_key,
                    "starred": _stored_router_starred(
                        base_url=base_url,
                        connections=existing,
                        endpoint_records=endpoint_records,
                    ),
                    "priority": priority,
                    "active": active,
                    "manually_disabled": False,
                    "last_error": last_error,
                }
            )
            save_llm_router_endpoint_records(resolved_layout, endpoint_records)
        except (ValidationError, ValueError) as exc:
            return _redirect_with_error(next_url, str(exc))
        return RedirectResponse(next_url, status_code=303)

    @app.post("/router/modal-workers/start")
    async def start_modal_worker(request: Request) -> RedirectResponse:
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        model = (parsed_form.get("modal_model", [""])[0] or "").strip() or DEFAULT_MODAL_WORKER_MODEL
        execution_mode = "vllm"
        raw_duration = (parsed_form.get("modal_duration_minutes", ["60"])[0] or "60").strip()
        try:
            duration_minutes = max(1, min(10080, int(raw_duration or "60")))
        except ValueError:
            return _redirect_with_error(next_url, "Modal duration must be an integer number of minutes.")
        if importlib.util.find_spec("modal") is None:
            return _redirect_with_error(next_url, "The `modal` package is not installed in llm_router.")
        launch_id = uuid4().hex
        safe_display_name = f"modal-{launch_id[:6]}"
        log_path = _modal_worker_log_path(resolved_layout, launch_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "modal_worker.py"),
            "--router-url",
            _router_control_base_url(str(shared_llm.base_url)),
            "--model",
            model,
            "--display-name",
            safe_display_name,
            "--duration-minutes",
            str(duration_minutes),
            "--launch-id",
            launch_id,
            "--max-concurrent",
            "8",
            "--execution-mode",
            execution_mode,
            "--state-path",
            str(_modal_worker_state_path(resolved_layout, launch_id)),
        ]
        env = os.environ.copy()
        env.update(load_dotenv_file(_router_env_file_path(resolved_layout)))
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(resolved_layout.project_root),
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    text=True,
                )
        except OSError as exc:
            return _redirect_with_error(next_url, f"Failed to launch Modal worker: {exc}")
        launches = _reconcile_modal_worker_launches(
            layout=resolved_layout,
            workers=worker_registry.list_workers(),
        )
        launches = [launch for launch in launches if launch.launch_id != launch_id]
        launches.insert(
            0,
            ModalWorkerLaunchRecord(
                launch_id=launch_id,
                display_name=safe_display_name,
                model=model,
                duration_minutes=duration_minutes,
                started_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=duration_minutes),
                pid=process.pid,
                worker_id=None,
                log_path=str(log_path),
                command=command,
                status=ModalWorkerLaunchStatus.starting,
                metadata={
                    "launcher": "modal_worker",
                    "execution_mode": execution_mode,
                },
            ),
        )
        save_modal_worker_launches(resolved_layout, launches)
        app.state.router_panel_cache = None
        return RedirectResponse(next_url, status_code=303)

    @app.post("/router/modal-workers/{launch_id}/stop")
    async def stop_modal_worker(request: Request, launch_id: str) -> RedirectResponse:
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        launches = _reconcile_modal_worker_launches(
            layout=resolved_layout,
            workers=worker_registry.list_workers(),
        )
        updated = False
        next_launches: list[ModalWorkerLaunchRecord] = []
        target_worker_id: str | None = None
        modal_app_id: str | None = None
        modal_stop_error: str | None = None
        for launch in launches:
            if launch.launch_id != launch_id:
                next_launches.append(launch)
                continue
            updated = True
            target_worker_id = launch.worker_id
            state_payload = _load_modal_worker_state(resolved_layout, launch.launch_id)
            modal_app_id = (
                _coerce_optional(launch.metadata.get("modal_app_id") if isinstance(launch.metadata, dict) else None)
                or _coerce_optional(state_payload.get("app_id"))
            )
            if _pid_is_alive(launch.pid):
                try:
                    os.killpg(launch.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            stopped_ok, modal_stop_error = _stop_modal_launch_app(
                layout=resolved_layout,
                launch_id=launch.launch_id,
                modal_app_id=modal_app_id,
            )
            next_metadata = dict(launch.metadata)
            if modal_app_id is not None:
                next_metadata["modal_app_id"] = modal_app_id
            next_launches.append(
                launch.model_copy(
                    update={
                        "status": ModalWorkerLaunchStatus.stopped if stopped_ok else ModalWorkerLaunchStatus.failed,
                        "worker_id": None,
                        "pid": None,
                        "last_error": modal_stop_error,
                        "metadata": next_metadata,
                    }
                )
            )
        if not updated:
            return _redirect_with_error(next_url, f"Modal worker launch not found: {launch_id}")
        save_modal_worker_launches(resolved_layout, next_launches)
        if target_worker_id:
            try:
                worker_registry.disconnect(target_worker_id, reason="stopped from UI")
            except KeyError:
                pass
            else:
                await _persist_worker_registry()
                await _sync_router_sources(refresh_health=False)
        app.state.router_panel_cache = None
        return RedirectResponse(next_url, status_code=303)

    @app.post("/router/vast/start")
    async def start_vast_instance(request: Request) -> RedirectResponse:
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        return _redirect_with_error(next_url, "Vast support has been removed from llm_router.")

    @app.post("/router/vast/stop")
    async def stop_vast_instance(request: Request) -> RedirectResponse:
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        return _redirect_with_error(next_url, "Vast support has been removed from llm_router.")

    @app.post("/router/connections/{connection_id}")
    async def update_router_connection(request: Request, connection_id: str) -> Response:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
            raw_model = payload.get("model")
            raw_value = payload.get("max_concurrent_requests")
            raw_priority = payload.get("priority")
            raw_api_key = payload.get("api_key")
            raw_supports_image_inputs = payload.get("supports_image_inputs")
            if not isinstance(raw_model, str) or not raw_model.strip():
                raise HTTPException(status_code=400, detail="model must be a non-empty string.")
            try:
                max_concurrent = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="max_concurrent_requests must be an integer.") from exc
            if raw_api_key is None:
                api_key = None
            elif isinstance(raw_api_key, str):
                api_key = raw_api_key.strip() or None
            else:
                raise HTTPException(status_code=400, detail="api_key must be a string when provided.")
            if raw_supports_image_inputs is None:
                supports_image_inputs: bool | None = None
            elif isinstance(raw_supports_image_inputs, bool):
                supports_image_inputs = raw_supports_image_inputs
            else:
                raise HTTPException(status_code=400, detail="supports_image_inputs must be a boolean when provided.")
            if raw_priority is None:
                priority: int | None = None
            else:
                try:
                    priority = int(raw_priority)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=400, detail="priority must be an integer.") from exc
            snapshot = dashboard_llm_router.snapshot()
            remaining: list[LLMRouterConnectionConfig] = []
            updated = False
            for item in _serialize_manual_router_connections(snapshot):
                if item.connection_id == connection_id:
                    remaining.append(
                        LLMRouterConnectionConfig(
                            connection_id=item.connection_id,
                            base_url=item.base_url,
                            model=raw_model.strip(),
                            source_kind=item.source_kind,
                            worker_id=item.worker_id,
                            api_key=api_key if raw_api_key is not None else item.api_key,
                            starred=item.starred,
                            priority=priority if priority is not None else item.priority,
                            max_concurrent_requests=max_concurrent,
                            supports_image_inputs=(
                                supports_image_inputs if supports_image_inputs is not None else item.supports_image_inputs
                            ),
                        )
                    )
                    updated = True
                else:
                    remaining.append(
                        LLMRouterConnectionConfig(
                            connection_id=item.connection_id,
                            base_url=item.base_url,
                            model=item.model,
                            source_kind=item.source_kind,
                            worker_id=item.worker_id,
                            api_key=item.api_key,
                            starred=item.starred,
                            priority=item.priority,
                            max_concurrent_requests=item.max_concurrent_requests,
                            supports_image_inputs=item.supports_image_inputs,
                        )
                    )
            if not updated:
                raise HTTPException(status_code=404, detail=f"Router connection not found: {connection_id}")
            manual_router_connections[:] = remaining
            await _sync_router_sources(refresh_health=True)
            return JSONResponse(dashboard_llm_router.snapshot().model_dump(mode="json"))

        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        replacement_model = (parsed_form.get("connection_model", [""])[0] or "").strip()
        try:
            max_concurrent = int(parsed_form.get("connection_max_concurrent", ["1"])[0] or "1")
        except ValueError:
            return _redirect_with_error(next_url, "Max concurrent must be an integer.")
        supports_image_inputs = bool(parsed_form.get("connection_supports_images"))
        if not replacement_model:
            return _redirect_with_error(next_url, "Model must be non-empty.")
        snapshot = dashboard_llm_router.snapshot()
        remaining: list[LLMRouterConnectionConfig] = []
        updated = False
        for item in _serialize_manual_router_connections(snapshot):
            if item.connection_id == connection_id:
                remaining.append(
                    LLMRouterConnectionConfig(
                        connection_id=item.connection_id,
                        base_url=item.base_url,
                        model=replacement_model,
                        source_kind=item.source_kind,
                        worker_id=item.worker_id,
                        api_key=item.api_key,
                        starred=item.starred,
                        priority=item.priority,
                        max_concurrent_requests=max_concurrent,
                        supports_image_inputs=supports_image_inputs,
                    )
                )
                updated = True
            else:
                remaining.append(
                    LLMRouterConnectionConfig(
                        connection_id=item.connection_id,
                        base_url=item.base_url,
                        model=item.model,
                        source_kind=item.source_kind,
                        worker_id=item.worker_id,
                        api_key=item.api_key,
                        starred=item.starred,
                        priority=item.priority,
                        max_concurrent_requests=item.max_concurrent_requests,
                        supports_image_inputs=item.supports_image_inputs,
                    )
                )
        if not updated:
            return _redirect_with_error(next_url, f"Router model not found: {connection_id}")
        try:
            manual_router_connections[:] = remaining
            await _sync_router_sources(refresh_health=True)
        except (TypeError, ValueError, ValidationError) as exc:
            return _redirect_with_error(next_url, str(exc))
        return RedirectResponse(next_url, status_code=303)

    @app.post("/router/urls/update")
    async def update_router_url(request: Request) -> RedirectResponse:
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        base_url = (parsed_form.get("connection_base_url", [""])[0] or "").strip()
        replacement_api_key = (parsed_form.get("connection_api_key", [""])[0] or "").strip()
        if not base_url:
            return _redirect_with_error(next_url, "Base URL must be non-empty.")
        endpoint_records = load_llm_router_endpoint_records(resolved_layout)
        snapshot = dashboard_llm_router.snapshot()
        replacement_api_key = replacement_api_key or _stored_router_api_key(
            base_url=base_url,
            connections=_serialize_manual_router_connections(snapshot),
            endpoint_records=endpoint_records,
        )
        remaining: list[LLMRouterConnectionConfig] = []
        updated = False
        for item in _serialize_manual_router_connections(snapshot):
            if item.base_url == base_url:
                remaining.append(
                    LLMRouterConnectionConfig(
                        connection_id=item.connection_id,
                        base_url=item.base_url,
                        model=item.model,
                        source_kind=item.source_kind,
                        worker_id=item.worker_id,
                        api_key=replacement_api_key,
                        starred=item.starred,
                        priority=item.priority,
                        max_concurrent_requests=item.max_concurrent_requests,
                        supports_image_inputs=item.supports_image_inputs,
                    )
                )
                updated = True
            else:
                remaining.append(
                    LLMRouterConnectionConfig(
                        connection_id=item.connection_id,
                        base_url=item.base_url,
                        model=item.model,
                        source_kind=item.source_kind,
                        worker_id=item.worker_id,
                        api_key=item.api_key,
                        starred=item.starred,
                        priority=item.priority,
                        max_concurrent_requests=item.max_concurrent_requests,
                        supports_image_inputs=item.supports_image_inputs,
                    )
                )
        if not updated:
            saved_endpoint = next((item for item in endpoint_records if item["base_url"] == base_url), None)
            if saved_endpoint is None:
                return _redirect_with_error(next_url, f"Router URL not found: {base_url}")
        manual_router_connections[:] = remaining
        await _sync_router_sources(refresh_health=True)
        active, last_error = await _probe_router_endpoint_health(base_url, replacement_api_key or None)
        endpoint_records = [item for item in endpoint_records if item["base_url"] != base_url]
        endpoint_records.append(
            {
                "base_url": base_url,
                "api_key": replacement_api_key,
                "starred": _stored_router_starred(
                    base_url=base_url,
                    connections=remaining,
                    endpoint_records=endpoint_records,
                ),
                "priority": _stored_router_priority(
                    base_url=base_url,
                    connections=remaining,
                    endpoint_records=endpoint_records,
                ),
                "active": active,
                "manually_disabled": False,
                "last_error": last_error,
            }
        )
        save_llm_router_endpoint_records(resolved_layout, endpoint_records)
        return RedirectResponse(next_url, status_code=303)

    @app.post("/router/urls/priority")
    async def update_router_url_priority(request: Request) -> RedirectResponse:
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        base_url = (parsed_form.get("connection_base_url", [""])[0] or "").strip()
        raw_priority = (parsed_form.get("connection_priority", ["2"])[0] or "2").strip()
        if not base_url:
            return _redirect_with_error(next_url, "Base URL must be non-empty.")
        try:
            priority = max(1, min(99, int(raw_priority)))
        except ValueError:
            return _redirect_with_error(next_url, "Priority must be an integer.")
        snapshot = dashboard_llm_router.snapshot()
        manual_snapshot = _serialize_manual_router_connections(snapshot)
        endpoint_records = load_llm_router_endpoint_records(resolved_layout)
        updated = False
        remaining = []
        for item in manual_snapshot:
            if item.base_url == base_url:
                remaining.append(item.model_copy(update={"priority": priority}))
                updated = True
            else:
                remaining.append(item)
        if updated:
            manual_router_connections[:] = remaining
            await _sync_router_sources(refresh_health=False)
        matched_record = next((item for item in endpoint_records if item["base_url"] == base_url), None)
        if matched_record is None:
            endpoint_records.append(
                {
                    "base_url": base_url,
                    "api_key": _stored_router_api_key(
                        base_url=base_url,
                        connections=remaining if updated else manual_snapshot,
                        endpoint_records=endpoint_records,
                    ),
                    "starred": _stored_router_starred(
                        base_url=base_url,
                        connections=remaining if updated else manual_snapshot,
                        endpoint_records=endpoint_records,
                    ),
                    "priority": priority,
                    "active": True,
                    "manually_disabled": False,
                    "last_error": None,
                }
            )
        else:
            matched_record["priority"] = priority
        save_llm_router_endpoint_records(resolved_layout, endpoint_records)
        return RedirectResponse(next_url, status_code=303)

    @app.post("/router/urls/{action}")
    async def mutate_router_url(request: Request, action: str) -> RedirectResponse:
        body_text = (await request.body()).decode("utf-8")
        parsed_form = parse_qs(body_text, keep_blank_values=True)
        next_url = parsed_form.get("next", ["/"])[0] or "/"
        base_url = (parsed_form.get("connection_base_url", [""])[0] or "").strip()
        if action not in {"enable", "disable", "delete", "star", "unstar"}:
            raise HTTPException(status_code=404, detail="Unknown router URL action.")
        if not base_url:
            return _redirect_with_error(next_url, "Base URL must be non-empty.")
        snapshot = dashboard_llm_router.snapshot()
        manual_snapshot = _serialize_manual_router_connections(snapshot)
        matching_ids = [item.connection_id for item in manual_snapshot if item.base_url == base_url]
        endpoint_records = load_llm_router_endpoint_records(resolved_layout)
        if not matching_ids and not any(item["base_url"] == base_url for item in endpoint_records):
            return _redirect_with_error(next_url, f"Router URL not found: {base_url}")
        if action == "delete":
            manual_router_connections[:] = [item for item in manual_snapshot if item.base_url != base_url]
            await _sync_router_sources(refresh_health=True)
            save_llm_router_endpoint_records(
                resolved_layout,
                [item for item in endpoint_records if item["base_url"] != base_url],
            )
            return RedirectResponse(next_url, status_code=303)
        if action in {"star", "unstar"}:
            starred = action == "star"
            remaining = [
                item.model_copy(update={"starred": starred if item.base_url == base_url else item.starred})
                for item in manual_snapshot
            ]
            manual_router_connections[:] = remaining
            await _sync_router_sources(refresh_health=False)
            matched_record = next((item for item in endpoint_records if item["base_url"] == base_url), None)
            if matched_record is None:
                endpoint_records.append(
                    {
                        "base_url": base_url,
                        "api_key": _stored_router_api_key(
                            base_url=base_url,
                            connections=remaining,
                            endpoint_records=endpoint_records,
                        ),
                        "starred": starred,
                        "priority": _stored_router_priority(
                            base_url=base_url,
                            connections=remaining,
                            endpoint_records=endpoint_records,
                        ),
                        "active": True,
                        "manually_disabled": False,
                        "last_error": None,
                    }
                )
            else:
                matched_record["starred"] = starred
            save_llm_router_endpoint_records(resolved_layout, endpoint_records)
            return RedirectResponse(next_url, status_code=303)
        disabled = action == "disable"
        for connection_id in matching_ids:
            await dashboard_llm_router.set_connection_manually_disabled(connection_id, disabled=disabled)
        if not disabled:
            await dashboard_llm_router.refresh_connection_health()
        await _persist_router_snapshot(layout=resolved_layout, router=dashboard_llm_router)
        matched_record = next((item for item in endpoint_records if item["base_url"] == base_url), None)
        if matched_record is not None:
            matched_record["manually_disabled"] = disabled
            matched_record["active"] = False if disabled else matched_record.get("active", True)
            matched_record["last_error"] = "Manually disabled" if disabled else None
            save_llm_router_endpoint_records(resolved_layout, endpoint_records)
        return RedirectResponse(next_url, status_code=303)

    @app.post("/router/connections/{connection_id}/disable")
    async def disable_router_connection(request: Request, connection_id: str) -> Response:
        content_type = request.headers.get("content-type", "")
        next_url = "/"
        if "application/json" not in content_type:
            body_text = (await request.body()).decode("utf-8")
            parsed_form = parse_qs(body_text, keep_blank_values=True)
            next_url = parsed_form.get("next", ["/"])[0] or "/"
        try:
            await dashboard_llm_router.set_connection_manually_disabled(connection_id, disabled=True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Router connection not found: {connection_id}") from exc
        await _persist_router_snapshot(layout=resolved_layout, router=dashboard_llm_router)
        if "application/json" not in content_type:
            return RedirectResponse(next_url, status_code=303)
        return JSONResponse(dashboard_llm_router.snapshot().model_dump(mode="json"))

    @app.post("/router/connections/{connection_id}/enable")
    async def enable_router_connection(request: Request, connection_id: str) -> Response:
        content_type = request.headers.get("content-type", "")
        next_url = "/"
        if "application/json" not in content_type:
            body_text = (await request.body()).decode("utf-8")
            parsed_form = parse_qs(body_text, keep_blank_values=True)
            next_url = parsed_form.get("next", ["/"])[0] or "/"
        try:
            await dashboard_llm_router.set_connection_manually_disabled(connection_id, disabled=False)
            await dashboard_llm_router.refresh_connection_health()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Router connection not found: {connection_id}") from exc
        await _persist_router_snapshot(layout=resolved_layout, router=dashboard_llm_router)
        if "application/json" not in content_type:
            return RedirectResponse(next_url, status_code=303)
        return JSONResponse(dashboard_llm_router.snapshot().model_dump(mode="json"))

    @app.post("/router/connections/{connection_id}/delete")
    async def delete_router_connection(request: Request, connection_id: str) -> Response:
        content_type = request.headers.get("content-type", "")
        next_url = "/"
        if "application/json" not in content_type:
            body_text = (await request.body()).decode("utf-8")
            parsed_form = parse_qs(body_text, keep_blank_values=True)
            next_url = parsed_form.get("next", ["/"])[0] or "/"
        snapshot = dashboard_llm_router.snapshot()
        manual_snapshot = _serialize_manual_router_connections(snapshot)
        remaining = [
            LLMRouterConnectionConfig(
                connection_id=item.connection_id,
                base_url=item.base_url,
                model=item.model,
                source_kind=item.source_kind,
                worker_id=item.worker_id,
                api_key=item.api_key,
                starred=item.starred,
                priority=item.priority,
                max_concurrent_requests=item.max_concurrent_requests,
                supports_image_inputs=item.supports_image_inputs,
            )
            for item in manual_snapshot
            if item.connection_id != connection_id
        ]
        if len(remaining) != len(manual_snapshot):
            manual_router_connections[:] = remaining
            await _sync_router_sources(refresh_health=True)
        if "application/json" not in content_type:
            return RedirectResponse(next_url, status_code=303)
        return JSONResponse(dashboard_llm_router.snapshot().model_dump(mode="json"))

    @app.post("/router/workers/{worker_id}/parallelism")
    async def set_worker_parallelism(request: Request, worker_id: str) -> Response:
        content_type = request.headers.get("content-type", "")
        next_url = "/"
        requested_limit: int | None = None
        if "application/json" in content_type:
            payload = WorkerConcurrencySettingsRequest.model_validate(await request.json())
            requested_limit = payload.max_concurrent_requests
        else:
            body_text = (await request.body()).decode("utf-8")
            parsed_form = parse_qs(body_text, keep_blank_values=True)
            next_url = parsed_form.get("next", ["/"])[0] or "/"
            raw_limit = (parsed_form.get("router_max_concurrent_requests", [""])[0] or "").strip()
            if raw_limit:
                try:
                    requested_limit = int(raw_limit)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail="router_max_concurrent_requests must be an integer",
                    ) from exc
        try:
            worker = worker_registry.set_router_max_concurrent_requests(
                worker_id,
                router_max_concurrent_requests=requested_limit,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await _persist_worker_registry()
        await _sync_router_sources(refresh_health=False)
        if "application/json" not in content_type:
            return RedirectResponse(next_url, status_code=303)
        return JSONResponse({"worker": worker.model_dump(mode="json")})

    @app.post("/router/workers/{worker_id}/priority")
    async def set_worker_priority(request: Request, worker_id: str) -> Response:
        content_type = request.headers.get("content-type", "")
        next_url = "/"
        if "application/json" in content_type:
            payload = RouterPrioritySettingsRequest.model_validate(await request.json())
            requested_priority = payload.priority
        else:
            body_text = (await request.body()).decode("utf-8")
            parsed_form = parse_qs(body_text, keep_blank_values=True)
            next_url = parsed_form.get("next", ["/"])[0] or "/"
            raw_priority = (parsed_form.get("router_priority", ["2"])[0] or "2").strip()
            try:
                requested_priority = max(1, min(99, int(raw_priority)))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="router_priority must be an integer") from exc
        try:
            worker = worker_registry.set_router_priority(
                worker_id,
                router_priority=requested_priority,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await _persist_worker_registry()
        await _sync_router_sources(refresh_health=False)
        if "application/json" not in content_type:
            return RedirectResponse(next_url, status_code=303)
        return JSONResponse({"worker": worker.model_dump(mode="json")})

    @app.get("/api/v1/workers")
    async def list_workers() -> JSONResponse:
        return JSONResponse(
            {"workers": [item.model_dump(mode="json") for item in worker_registry.list_workers()]}
        )

    @app.get("/api/v1/workers/{worker_id}")
    async def get_worker(worker_id: str) -> JSONResponse:
        try:
            worker = worker_registry.get_worker(worker_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(worker.model_dump(mode="json"))

    @app.post("/api/v1/workers/{worker_id}/settings/concurrency")
    async def set_worker_concurrency_settings(
        worker_id: str,
        payload: WorkerConcurrencySettingsRequest,
    ) -> JSONResponse:
        try:
            worker = worker_registry.set_router_max_concurrent_requests(
                worker_id,
                router_max_concurrent_requests=payload.max_concurrent_requests,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await _persist_worker_registry()
        await _sync_router_sources(refresh_health=False)
        return JSONResponse({"worker": worker.model_dump(mode="json")})

    @app.post("/api/v1/workers/{worker_id}/settings/priority")
    async def set_worker_priority_settings(
        worker_id: str,
        payload: RouterPrioritySettingsRequest,
    ) -> JSONResponse:
        try:
            worker = worker_registry.set_router_priority(
                worker_id,
                router_priority=payload.priority,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await _persist_worker_registry()
        await _sync_router_sources(refresh_health=False)
        return JSONResponse({"worker": worker.model_dump(mode="json")})

    @app.post("/api/v1/workers/register")
    async def register_worker(payload: LLMWorkerRegisterRequest) -> JSONResponse:
        worker = worker_registry.register(payload)
        metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
        if metadata.get("launcher") == "modal_worker":
            requested_priority = metadata.get("desired_router_priority", 3)
            try:
                worker = worker_registry.set_router_priority(
                    worker.worker_id,
                    router_priority=max(1, min(99, int(requested_priority))),
                )
            except (KeyError, TypeError, ValueError):
                pass
        await _persist_worker_registry()
        await _sync_router_sources(refresh_health=True)
        return JSONResponse(
            {
                "worker": worker.model_dump(mode="json"),
                "connection_id": f"worker:{worker.worker_id}",
            }
        )

    @app.post("/api/v1/workers/{worker_id}/heartbeat")
    async def heartbeat_worker(
        worker_id: str,
        payload: LLMWorkerHeartbeatRequest | None = None,
    ) -> JSONResponse:
        try:
            worker = worker_registry.heartbeat(worker_id, payload or LLMWorkerHeartbeatRequest())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await _persist_worker_registry()
        await _sync_router_sources(refresh_health=False)
        return JSONResponse({"worker": worker.model_dump(mode="json")})

    @app.post("/api/v1/workers/{worker_id}/disconnect")
    async def disconnect_worker(
        worker_id: str,
        payload: LLMWorkerDisconnectRequest | None = None,
    ) -> JSONResponse:
        try:
            worker = worker_registry.disconnect(worker_id, reason=(payload.reason if payload else None))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await _persist_worker_registry()
        await _sync_router_sources(refresh_health=False)
        return JSONResponse({"worker": worker.model_dump(mode="json")})

    @app.post("/api/v1/workers/{worker_id}/requests/poll")
    async def poll_worker_requests(
        worker_id: str,
        payload: WorkerRequestPollRequest | None = None,
    ) -> JSONResponse:
        try:
            worker = worker_registry.get_worker(worker_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        effective_limit = worker.effective_max_concurrent_requests
        requests = await worker_request_queue.poll(
            worker_id=worker_id,
            limit=min(
                payload.max_requests if payload is not None else effective_limit,
                effective_limit,
            ),
        )
        return JSONResponse({"requests": requests})

    @app.post("/api/v1/workers/{worker_id}/requests/{request_id}/complete")
    async def complete_worker_request(
        worker_id: str,
        request_id: str,
        payload: WorkerRequestCompleteRequest,
    ) -> JSONResponse:
        try:
            await worker_request_queue.complete(
                worker_id=worker_id,
                request_id=request_id,
                response_body=payload.response_body,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown worker request {request_id}") from exc
        return JSONResponse({"ok": True})

    @app.post("/api/v1/workers/{worker_id}/requests/{request_id}/fail")
    async def fail_worker_request(
        worker_id: str,
        request_id: str,
        payload: WorkerRequestFailRequest,
    ) -> JSONResponse:
        try:
            await worker_request_queue.fail(
                worker_id=worker_id,
                request_id=request_id,
                error=payload.error,
                status_code=payload.status_code,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown worker request {request_id}") from exc
        return JSONResponse({"ok": True})

    async def _router_models_response() -> JSONResponse:
        snapshot = dashboard_llm_router.snapshot()
        if not any(item.active for item in snapshot.connections):
            raise HTTPException(status_code=503, detail="Dashboard LLM router has no active backend connections.")
        return JSONResponse(await dashboard_llm_router.list_models_payload())

    async def _router_chat_completions_response(request: Request) -> JSONResponse:
        started_at = datetime.now(timezone.utc)
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Expected JSON request body.") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object request body.")
        try:
            result = await dashboard_llm_router.chat_completion(payload)
        except UnknownLLMModelError as exc:
            _log_router_request_error(layout=resolved_layout, payload=payload, error=exc)
            await asyncio.to_thread(
                _record_router_request,
                request=request,
                payload=payload,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="failed",
                http_status_code=404,
                error_message=str(exc),
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except NoActiveLLMConnectionError as exc:
            _update_router_connection_outage_state()
            _log_router_request_error(layout=resolved_layout, payload=payload, error=exc)
            await asyncio.to_thread(
                _record_router_request,
                request=request,
                payload=payload,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="failed",
                http_status_code=503,
                error_message=str(exc),
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            error_detail = DashboardLLMRouter._format_upstream_error(exc)
            _log_router_request_error(layout=resolved_layout, payload=payload, error=exc)
            await asyncio.to_thread(
                _record_router_request,
                request=request,
                payload=payload,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="failed",
                http_status_code=502,
                error_message=error_detail,
            )
            raise HTTPException(status_code=502, detail=error_detail) from exc
        except httpx.HTTPError as exc:
            _log_router_request_error(layout=resolved_layout, payload=payload, error=exc)
            await asyncio.to_thread(
                _record_router_request,
                request=request,
                payload=payload,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="failed",
                http_status_code=503,
                error_message=str(exc),
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except WorkerRequestError as exc:
            _log_router_request_error(layout=resolved_layout, payload=payload, error=exc)
            status_code = exc.status_code or 502
            if status_code < 400 or status_code > 599:
                status_code = 502
            await asyncio.to_thread(
                _record_router_request,
                request=request,
                payload=payload,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="failed",
                http_status_code=status_code,
                error_message=str(exc),
            )
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        except ValueError as exc:
            _log_router_request_error(layout=resolved_layout, payload=payload, error=exc)
            await asyncio.to_thread(
                _record_router_request,
                request=request,
                payload=payload,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="failed",
                http_status_code=400,
                error_message=str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        completed_at = datetime.now(timezone.utc)
        await asyncio.to_thread(
            _record_router_request,
            request=request,
            payload=payload,
            started_at=started_at,
            completed_at=completed_at,
            status="completed",
            http_status_code=200,
            response_body=result.body,
        )
        await _persist_router_snapshot(layout=resolved_layout, router=dashboard_llm_router)
        return JSONResponse(
            result.body,
            headers={
                "X-LLM-Connection-Id": result.connection_id,
                "X-LLM-Server": result.base_url,
                "X-LLM-Model": result.model,
            },
        )

    @app.get("/v1/models")
    async def router_models(request: Request) -> JSONResponse:
        if _request_uses_public_router_host(request, resolved_layout):
            _require_public_api_key(request, resolved_layout)
        return await _router_models_response()

    @app.get(f"{ROUTER_PUBLIC_API_PATH}/models")
    async def router_public_models(request: Request) -> JSONResponse:
        _require_public_api_key(request, resolved_layout)
        return await _router_models_response()

    @app.post("/v1/chat/completions")
    async def router_chat_completions(request: Request) -> JSONResponse:
        if _request_uses_public_router_host(request, resolved_layout):
            _require_public_api_key(request, resolved_layout)
        return await _router_chat_completions_response(request)

    @app.post(f"{ROUTER_PUBLIC_API_PATH}/chat/completions")
    async def router_public_chat_completions(request: Request) -> JSONResponse:
        _require_public_api_key(request, resolved_layout)
        return await _router_chat_completions_response(request)

    return app


app = create_router_app()
