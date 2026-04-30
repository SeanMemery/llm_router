from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlparse
import math

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

GLOBAL_MIN_COMPLETION_TOKENS = 8192


class LLMRouterError(RuntimeError):
    """Base error for dashboard-owned router failures."""


class NoActiveLLMConnectionError(LLMRouterError):
    """Raised when no active backend connections are available."""


class MissingUsageTelemetryError(LLMRouterError):
    """Raised when an upstream response omits required token-usage fields."""


class UnknownLLMModelError(LLMRouterError):
    """Raised when a request asks for a model that is not configured."""


class LLMRouterConnectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    base_url: str
    model: str
    source_kind: Literal["manual", "worker"] = "manual"
    worker_id: str | None = None
    transport_mode: Literal["direct-endpoint", "pull-request"] = "direct-endpoint"
    api_key: str | None = None
    starred: bool = False
    priority: int = Field(default=2, ge=1, le=99)
    max_concurrent_requests: int = Field(ge=1)
    supports_image_inputs: bool = True

    @model_validator(mode="after")
    def validate_fields(self) -> "LLMRouterConnectionConfig":
        if not self.connection_id.strip():
            raise ValueError("connection_id must be non-empty")
        if not self.base_url.strip():
            raise ValueError("base_url must be non-empty")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.source_kind == "worker":
            worker_id = (self.worker_id or "").strip()
            if not worker_id:
                raise ValueError("worker_id must be provided for worker connections")
            self.worker_id = worker_id
        else:
            self.worker_id = None
            self.transport_mode = "direct-endpoint"
        if self.api_key is not None:
            api_key = self.api_key.strip()
            if not api_key:
                raise ValueError("api_key must be non-empty when provided")
            self.api_key = api_key
        return self


class LLMRouterTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_request_seconds: float = 0.0
    total_output_tokens_per_second: float = 0.0
    total_concurrent_requests_seen: int = 0
    recent_request_samples: list[dict[str, float | int]] = Field(default_factory=list)

    @property
    def average_request_seconds(self) -> float | None:
        if self.request_count < 1:
            return None
        return self.total_request_seconds / self.request_count

    @property
    def average_output_tokens_per_second(self) -> float | None:
        samples = self.recent_request_samples[-25:]
        if not samples:
            return None
        total_completion_tokens = 0.0
        total_effective_seconds = 0.0
        for sample in samples:
            completion_tokens = float(sample.get("completion_tokens") or 0.0)
            request_seconds = max(float(sample.get("request_seconds") or 0.0), 1e-9)
            total_completion_tokens += completion_tokens
            total_effective_seconds += request_seconds
        if total_effective_seconds <= 0.0:
            return None
        return total_completion_tokens / total_effective_seconds

    @property
    def average_recent_request_seconds(self) -> float | None:
        samples = self.recent_request_samples[-25:]
        if not samples:
            return None
        total = 0.0
        for sample in samples:
            total += max(float(sample.get("request_seconds") or 0.0), 0.0)
        return total / len(samples)

    @property
    def average_context_tokens(self) -> float | None:
        if self.request_count < 1:
            return None
        return self.total_prompt_tokens / self.request_count

    @property
    def average_concurrent_requests(self) -> float | None:
        if self.request_count < 1:
            return None
        return self.total_concurrent_requests_seen / self.request_count

    def record_request(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        request_duration_seconds: float,
        concurrent_requests_seen: int,
    ) -> None:
        tokens_per_second = completion_tokens / max(request_duration_seconds, 1e-9)
        self.request_count += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_request_seconds += request_duration_seconds
        self.total_output_tokens_per_second += tokens_per_second
        self.total_concurrent_requests_seen += concurrent_requests_seen
        self.recent_request_samples.append(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "request_seconds": request_duration_seconds,
                "concurrent_requests_seen": concurrent_requests_seen,
                "tokens_per_second": tokens_per_second,
            }
        )
        if len(self.recent_request_samples) > 200:
            self.recent_request_samples = self.recent_request_samples[-200:]


class LLMRouterConnectionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    base_url: str
    model: str
    source_kind: Literal["manual", "worker"] = "manual"
    worker_id: str | None = None
    transport_mode: Literal["direct-endpoint", "pull-request"] = "direct-endpoint"
    canonical_model: str | None = None
    api_key: str | None = None
    starred: bool = False
    priority: int = Field(default=2, ge=1, le=99)
    max_concurrent_requests: int
    supports_image_inputs: bool = True
    manually_disabled: bool = False
    active: bool = True
    in_flight_requests: int = 0
    waiting_requests: int = 0
    last_error: str | None = None
    usage_percentage: float | None = None
    max_context_tokens: int | None = None
    telemetry: LLMRouterTelemetry = Field(default_factory=LLMRouterTelemetry)


class LLMRouterSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connections: list[LLMRouterConnectionSnapshot] = Field(default_factory=list)
    waiting_requests: int = 0


class LLMRouterCompletionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    body: dict[str, Any]
    connection_id: str
    base_url: str
    model: str


def normalize_canonical_model_name(name: str) -> str:
    """
    Produce a loose canonical model name so that close variants (quantization suffixes,
    file extensions, minor separators) match across backends.
    """
    if not name:
        return ""
    text = name.lower().strip()
    # drop pricing/plan suffixes like :free or :pro
    if ":" in text:
        text = text.split(":", 1)[0]
    # drop real file extensions (not version dots)
    if "." in text:
        head, ext = text.rsplit(".", 1)
        if ext.lower() in {"gguf", "bin", "pt", "safetensors", "json", "yaml"}:
            text = head
    # keep only final path segment (provider prefixes vary)
    if "/" in text:
        text = text.split("/")[-1]
    # unify separators
    text = text.replace("/", "-")
    # remove common quantization suffixes at end (e.g., -q4_k_m, -q8, -int4)
    tokens = [t for t in text.replace("_", "-").split("-") if t]
    filtered: list[str] = []
    drop_tokens = {"k", "m", "fp16", "bf16", "ud"}
    for token in tokens:
        if token.startswith("q") and len(token) >= 2 and token[1].isdigit():
            continue
        if token.startswith("int") and len(token) >= 4 and token[3:].isdigit():
            continue
        if token in {"gguf", "bin", "pt", "safetensors"}:
            continue
        if token in drop_tokens:
            continue
        filtered.append(token)
    if not filtered:
        filtered = tokens
    return "-".join(filtered)


@dataclass
class _ConnectionRuntime:
    config: LLMRouterConnectionConfig
    client: httpx.AsyncClient
    manually_disabled: bool = False
    active: bool = True
    in_flight_requests: int = 0
    last_error: str | None = None
    health_probe_failures: int = 0
    last_healthy_at: float | None = None
    observed_since: float = 0.0
    busy_seconds_total: float = 0.0
    busy_window_started_at: float | None = None
    telemetry: LLMRouterTelemetry = field(default_factory=LLMRouterTelemetry)
    canonical_model: str = ""
    max_context_tokens: int | None = None


class DashboardLLMRouter:
    def __init__(
        self,
        *,
        connections: list[LLMRouterConnectionConfig],
        timeout_seconds: float | None = None,
        no_active_connection_timeout_seconds: float = 600.0,
        transient_failure_retries: int = 2,
        transient_failure_backoff_seconds: float = 1.0,
        health_probe_failure_threshold: int = 3,
        health_probe_grace_seconds: float = 15.0,
        initial_snapshot: LLMRouterSnapshot | None = None,
        client_factory: Callable[
            [LLMRouterConnectionConfig, float | None], httpx.AsyncClient
        ]
        | None = None,
        worker_request_handler: Callable[[LLMRouterConnectionConfig, dict[str, Any]], Awaitable[dict[str, Any]]]
        | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._no_active_connection_timeout_seconds = no_active_connection_timeout_seconds
        self._transient_failure_retries = max(0, int(transient_failure_retries))
        self._transient_failure_backoff_seconds = max(
            0.0, float(transient_failure_backoff_seconds)
        )
        self._health_probe_failure_threshold = max(1, int(health_probe_failure_threshold))
        self._health_probe_grace_seconds = max(0.0, float(health_probe_grace_seconds))
        self._client_factory = client_factory or self._default_client_factory
        self._worker_request_handler = worker_request_handler
        self._time_fn = time_fn or time.perf_counter
        self._condition = asyncio.Condition()
        self._waiting_requests = 0
        snapshot_by_id = {
            item.connection_id: item for item in (initial_snapshot or LLMRouterSnapshot()).connections
        }
        self._connections: list[_ConnectionRuntime] = []
        for config in connections:
            prior = snapshot_by_id.get(config.connection_id)
            canonical = normalize_canonical_model_name(config.model)
            runtime = _ConnectionRuntime(
                config=config,
                client=self._client_factory(config, timeout_seconds),
                manually_disabled=prior.manually_disabled if prior is not None else False,
                active=prior.active if prior is not None else True,
                in_flight_requests=0,
                last_error=prior.last_error if prior is not None else None,
                last_healthy_at=None,
                observed_since=self._time_fn(),
                telemetry=prior.telemetry.model_copy(deep=True)
                if prior is not None
                else LLMRouterTelemetry(),
                canonical_model=canonical,
                max_context_tokens=prior.max_context_tokens if prior is not None else None,
            )
            self._connections.append(runtime)

    @staticmethod
    def _default_client_factory(
        config: LLMRouterConnectionConfig, timeout_seconds: float | None
    ) -> httpx.AsyncClient:
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        base_url = (
            "http://127.0.0.1"
            if DashboardLLMRouter._uses_pull_worker_transport(config)
            else config.base_url.rstrip("/")
        )
        return httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            headers=headers,
        )

    async def aclose(self) -> None:
        for runtime in self._connections:
            await runtime.client.aclose()

    def snapshot(self) -> LLMRouterSnapshot:
        now = self._time_fn()
        return LLMRouterSnapshot(
            waiting_requests=self._waiting_requests,
            connections=[
                LLMRouterConnectionSnapshot(
                    connection_id=runtime.config.connection_id,
                    base_url=runtime.config.base_url,
                    model=runtime.config.model,
                    source_kind=runtime.config.source_kind,
                    worker_id=runtime.config.worker_id,
                    transport_mode=runtime.config.transport_mode,
                    canonical_model=runtime.canonical_model,
                    api_key=runtime.config.api_key,
                    starred=runtime.config.starred,
                    priority=runtime.config.priority,
                    max_concurrent_requests=runtime.config.max_concurrent_requests,
                    supports_image_inputs=runtime.config.supports_image_inputs,
                    manually_disabled=runtime.manually_disabled,
                    active=runtime.active,
                    in_flight_requests=runtime.in_flight_requests,
                    waiting_requests=self._waiting_requests,
                    last_error=runtime.last_error,
                    usage_percentage=self._runtime_usage_percentage(runtime, now=now),
                    max_context_tokens=runtime.max_context_tokens,
                    telemetry=runtime.telemetry.model_copy(deep=True),
                )
                for runtime in self._connections
            ],
        )

    async def replace_connections(
        self,
        connections: list[LLMRouterConnectionConfig],
        *,
        preserve_snapshot: bool = True,
    ) -> None:
        previous = self.snapshot() if preserve_snapshot else None
        snapshot_by_id = {
            item.connection_id: item for item in (previous or LLMRouterSnapshot()).connections
        }
        prior_by_id = {runtime.config.connection_id: runtime for runtime in self._connections}
        next_connections: list[_ConnectionRuntime] = []
        clients_to_close: list[httpx.AsyncClient] = []
        for config in connections:
            prior = snapshot_by_id.get(config.connection_id)
            existing_runtime = prior_by_id.pop(config.connection_id, None)
            canonical = normalize_canonical_model_name(config.model)
            if existing_runtime is not None and self._can_reuse_runtime(existing_runtime.config, config):
                existing_runtime.config = config
                existing_runtime.canonical_model = canonical
                if prior is not None:
                    existing_runtime.manually_disabled = prior.manually_disabled
                    existing_runtime.active = prior.active
                    existing_runtime.last_error = prior.last_error
                    existing_runtime.telemetry = prior.telemetry.model_copy(deep=True)
                next_connections.append(existing_runtime)
                continue
            if existing_runtime is not None:
                clients_to_close.append(existing_runtime.client)
            next_connections.append(
                _ConnectionRuntime(
                    config=config,
                    client=self._client_factory(config, self._timeout_seconds),
                    manually_disabled=prior.manually_disabled if prior is not None else False,
                    active=prior.active if prior is not None else True,
                    in_flight_requests=0,
                    last_error=prior.last_error if prior is not None else None,
                    last_healthy_at=None,
                    observed_since=self._time_fn(),
                    telemetry=prior.telemetry.model_copy(deep=True)
                    if prior is not None
                    else LLMRouterTelemetry(),
                    canonical_model=canonical,
                    max_context_tokens=prior.max_context_tokens if prior is not None else None,
                )
            )
        clients_to_close.extend(runtime.client for runtime in prior_by_id.values())
        async with self._condition:
            self._connections = next_connections
            self._condition.notify_all()
        for client in clients_to_close:
            await client.aclose()

    async def set_connection_active(
        self,
        connection_id: str,
        *,
        active: bool,
        last_error: str | None = None,
    ) -> None:
        async with self._condition:
            runtime = self._runtime_for_id(connection_id)
            if runtime.manually_disabled and active:
                runtime.active = False
                runtime.last_error = "Manually disabled"
                self._condition.notify_all()
                return
            runtime.active = active
            runtime.last_error = last_error
            self._condition.notify_all()

    async def set_connection_manually_disabled(
        self,
        connection_id: str,
        *,
        disabled: bool,
    ) -> None:
        async with self._condition:
            runtime = self._runtime_for_id(connection_id)
            runtime.manually_disabled = disabled
            if disabled:
                runtime.active = False
                runtime.last_error = "Manually disabled"
            else:
                runtime.last_error = None
            self._condition.notify_all()

    async def refresh_connection_health(self) -> None:
        tasks = [self._probe_runtime_health(runtime) for runtime in self._connections]
        if tasks:
            await asyncio.gather(*tasks)

    async def list_models_payload(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        available_connections = [
            connection
            for connection in snapshot.connections
            if connection.active and not connection.manually_disabled
        ]
        return {
            "object": "list",
            "data": [
                {
                    "id": connection.model,
                    "object": "model",
                    "owned_by": "dashboard-llm-router",
                    "metadata": {
                        "connection_id": connection.connection_id,
                        "base_url": connection.base_url,
                        "source_kind": connection.source_kind,
                        "worker_id": connection.worker_id,
                        "transport_mode": connection.transport_mode,
                        "starred": connection.starred,
                        "priority": connection.priority,
                        "active": connection.active,
                        "manually_disabled": connection.manually_disabled,
                        "supports_image_inputs": connection.supports_image_inputs,
                        "max_context_tokens": connection.max_context_tokens,
                    },
                }
                for connection in available_connections
            ],
        }

    async def chat_completion(self, payload: dict[str, Any]) -> LLMRouterCompletionResult:
        if payload.get("stream"):
            raise ValueError("Dashboard LLM router only supports non-streaming chat completions")
        outbound_payload = dict(payload)
        outbound_payload["stream"] = False
        requested_model = payload.get("model")
        if requested_model is not None:
            if not isinstance(requested_model, str):
                raise ValueError("model must be a string when provided")
            requested_model = requested_model.strip() or None
            canonical_requested = (
                normalize_canonical_model_name(requested_model) if requested_model else None
            )
            if requested_model is not None and not any(
                runtime.canonical_model == canonical_requested
                or runtime.config.model == requested_model
                for runtime in self._connections
            ):
                raise UnknownLLMModelError(
                    f"Requested model is not configured: {requested_model} (canonical {canonical_requested})"
                )
        else:
            requested_model = None
            canonical_requested = None
        requires_image_inputs = self._payload_requires_image_inputs(outbound_payload)
        required_total_tokens = self._estimate_required_total_tokens(outbound_payload)
        canonical_requested = (
            normalize_canonical_model_name(requested_model) if requested_model else None
        )
        if not self._has_any_capacity_candidate(
            requires_image_inputs=requires_image_inputs,
            requested_model=requested_model or canonical_requested,
            required_total_tokens=required_total_tokens,
        ):
            known_max_context = self._max_known_context_tokens(
                requires_image_inputs=requires_image_inputs,
                requested_model=requested_model or canonical_requested,
            )
            if known_max_context is not None:
                raise NoActiveLLMConnectionError(
                    "Dashboard LLM router has no active backend connection with sufficient context "
                    f"(estimated_required_tokens={required_total_tokens}, max_available_context={known_max_context})."
                )
        attempts = max(
            1 + self._transient_failure_retries,
            self._active_runtime_count(
                requires_image_inputs=requires_image_inputs,
                requested_model=requested_model or canonical_requested,
                required_total_tokens=required_total_tokens,
            ),
        )
        last_exc: Exception | None = None
        failed_connection_ids: set[str] = set()
        for attempt_index in range(attempts):
            runtime = await self._acquire_runtime(
                avoid_connection_ids=failed_connection_ids,
                requires_image_inputs=requires_image_inputs,
                requested_model=requested_model or canonical_requested,
                required_total_tokens=required_total_tokens,
            )
            concurrent_requests_seen = runtime.in_flight_requests
            request_started_at = self._time_fn()
            outbound_payload["model"] = runtime.config.model
            request_payload = self._normalize_chat_payload_for_connection(
                outbound_payload,
                runtime.config,
            )
            try:
                response = await runtime.client.post(
                    self._relative_connection_path(runtime.config, "chat/completions"),
                    json=request_payload,
                ) if not self._uses_pull_worker_transport(runtime.config) else None
                if self._uses_pull_worker_transport(runtime.config):
                    if self._worker_request_handler is None:
                        raise RuntimeError("Pull-request worker transport is not configured")
                    body = await self._worker_request_handler(runtime.config, request_payload)
                    if not isinstance(body, dict):
                        raise ValueError("Expected JSON object from pull-request worker")
                else:
                    assert response is not None
                    response.raise_for_status()
                    body = response.json()
                    if not isinstance(body, dict):
                        raise ValueError("Expected JSON object from upstream chat-completions endpoint")
                prompt_tokens, completion_tokens = self._extract_usage(body.get("usage"))
                request_duration_seconds = max(self._time_fn() - request_started_at, 1e-9)
                runtime.telemetry.record_request(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    request_duration_seconds=request_duration_seconds,
                    concurrent_requests_seen=concurrent_requests_seen,
                )
                runtime.health_probe_failures = 0
                runtime.last_healthy_at = self._time_fn()
                runtime.active = True
                runtime.last_error = None
                body["model"] = runtime.config.model
                router_metadata = body.get("router")
                if not isinstance(router_metadata, dict):
                    router_metadata = {}
                router_metadata["connection_id"] = runtime.config.connection_id
                router_metadata["base_url"] = runtime.config.base_url
                router_metadata["selected_model"] = runtime.config.model
                router_metadata["requested_model"] = requested_model
                router_metadata["requested_canonical_model"] = canonical_requested
                router_metadata["source_kind"] = runtime.config.source_kind
                router_metadata["transport_mode"] = runtime.config.transport_mode
                router_metadata["worker_id"] = runtime.config.worker_id
                router_metadata["concurrent_requests_seen"] = concurrent_requests_seen
                router_metadata["request_duration_seconds"] = round(request_duration_seconds, 6)
                body["router"] = router_metadata
                return LLMRouterCompletionResult(
                    body=body,
                    connection_id=runtime.config.connection_id,
                    base_url=runtime.config.base_url,
                    model=runtime.config.model,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if self._uses_pull_worker_transport(runtime.config):
                    runtime.last_error = self._format_upstream_error(exc)
                    failed_connection_ids.add(runtime.config.connection_id)
                    if (
                        attempt_index + 1 < attempts
                        and self._has_alternative_active_runtime(
                            failed_connection_ids,
                            requires_image_inputs=requires_image_inputs,
                            requested_model=requested_model,
                            required_total_tokens=required_total_tokens,
                        )
                    ):
                        continue
                    raise
                if isinstance(exc, httpx.HTTPStatusError):
                    runtime.last_error = self._format_upstream_error(exc)
                    failed_connection_ids.add(runtime.config.connection_id)
                    if self._should_deactivate_runtime_after_status(exc):
                        runtime.active = False
                        runtime.last_healthy_at = None
                    if self._is_retryable_upstream_status(exc):
                        if attempt_index + 1 < attempts:
                            await asyncio.sleep(
                                self._transient_backoff_for_attempt(attempt_index)
                            )
                            continue
                    elif (
                        attempt_index + 1 < attempts
                        and self._has_alternative_active_runtime(
                            failed_connection_ids,
                            requires_image_inputs=requires_image_inputs,
                            requested_model=requested_model,
                            required_total_tokens=required_total_tokens,
                        )
                    ):
                        continue
                elif isinstance(exc, httpx.HTTPError):
                    runtime.last_error = self._format_upstream_error(exc)
                    failed_connection_ids.add(runtime.config.connection_id)
                    if attempt_index + 1 < attempts:
                        await asyncio.sleep(
                            self._transient_backoff_for_attempt(attempt_index)
                        )
                        continue
                raise
            finally:
                await self._release_runtime(runtime)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Dashboard LLM router exhausted retries without a captured exception")

    async def _probe_runtime_health(self, runtime: _ConnectionRuntime) -> None:
        if runtime.manually_disabled:
            await self.set_connection_active(
                runtime.config.connection_id,
                active=False,
                last_error="Manually disabled",
            )
            return
        if self._uses_pull_worker_transport(runtime.config):
            runtime.health_probe_failures = 0
            runtime.last_healthy_at = self._time_fn()
            await self.set_connection_active(
                runtime.config.connection_id,
                active=True,
                last_error=runtime.last_error,
            )
            return
        try:
            response = await runtime.client.get(
                self._relative_connection_path(runtime.config, "models"),
                timeout=5.0,
            )
            response.raise_for_status()
            body = response.json()
            if not self._models_probe_body_has_models_list(body):
                raise RuntimeError("Upstream health probe did not return a models list")
            runtime.max_context_tokens = self._infer_runtime_context_tokens(runtime, body)
            runtime.health_probe_failures = 0
            runtime.last_healthy_at = self._time_fn()
            await self.set_connection_active(runtime.config.connection_id, active=True, last_error=None)
        except Exception as exc:  # noqa: BLE001
            if self._should_trust_recent_chat_success_for_health_probe(runtime, exc):
                runtime.health_probe_failures = 0
                runtime.active = True
                await self.set_connection_active(
                    runtime.config.connection_id,
                    active=True,
                    last_error=None,
                )
                return
            runtime.health_probe_failures += 1
            recently_healthy = (
                runtime.last_healthy_at is not None
                and (self._time_fn() - runtime.last_healthy_at) < self._health_probe_grace_seconds
            )
            if runtime.active and (
                runtime.in_flight_requests > 0
                or runtime.health_probe_failures < self._health_probe_failure_threshold
                or recently_healthy
            ):
                await self.set_connection_active(
                    runtime.config.connection_id,
                    active=True,
                    last_error=self._format_upstream_error(exc),
                )
                return
            await self.set_connection_active(
                runtime.config.connection_id,
                active=False,
                last_error=self._format_upstream_error(exc),
            )

    @staticmethod
    def _should_trust_recent_chat_success_for_health_probe(
        runtime: _ConnectionRuntime,
        exc: Exception,
    ) -> bool:
        if runtime.telemetry.request_count < 1:
            return False
        if not isinstance(exc, httpx.HTTPStatusError):
            return False
        status_code = exc.response.status_code
        if status_code not in {401, 403}:
            return False
        request = exc.request
        return request is not None and request.url.path.endswith("/models")

    @staticmethod
    def _models_probe_body_has_models_list(body: object) -> bool:
        if isinstance(body, list):
            return True
        if isinstance(body, dict) and isinstance(body.get("data"), list):
            return True
        return False

    async def _acquire_runtime(
        self,
        *,
        avoid_connection_ids: set[str] | None = None,
        requires_image_inputs: bool = False,
        requested_model: str | None = None,
        required_total_tokens: int | None = None,
    ) -> _ConnectionRuntime:
        no_active_deadline = self._time_fn() + self._no_active_connection_timeout_seconds
        avoided = set(avoid_connection_ids or ())
        while True:
            async with self._condition:
                available = self._available_runtimes(
                    requires_image_inputs=requires_image_inputs,
                    requested_model=requested_model,
                    required_total_tokens=required_total_tokens,
                )
                if available:
                    preferred = [
                        runtime
                        for runtime in available
                        if runtime.config.connection_id not in avoided
                    ]
                    candidates = preferred or available
                    runtime = min(
                        candidates,
                        key=lambda item: (
                            item.config.priority,
                            item.in_flight_requests / item.config.max_concurrent_requests,
                            item.in_flight_requests,
                            0 if item.config.starred else 1,
                            item.config.connection_id,
                        ),
                    )
                    if runtime.in_flight_requests == 0:
                        runtime.busy_window_started_at = self._time_fn()
                    runtime.in_flight_requests += 1
                    return runtime
                self._waiting_requests += 1
                try:
                    if any(
                        runtime.active
                        and (runtime.config.supports_image_inputs or not requires_image_inputs)
                        and (requested_model is None or runtime.config.model == requested_model)
                        and self._runtime_meets_context_requirement(
                            runtime,
                            required_total_tokens=required_total_tokens,
                        )
                        for runtime in self._connections
                    ):
                        await self._condition.wait()
                        no_active_deadline = (
                            self._time_fn() + self._no_active_connection_timeout_seconds
                        )
                    else:
                        remaining_seconds = no_active_deadline - self._time_fn()
                        if remaining_seconds <= 0:
                            raise NoActiveLLMConnectionError(
                                (
                                    f"Dashboard LLM router had no active backend connections for model {requested_model!r} for 10 minutes"
                                    if requested_model is not None
                                    else "Dashboard LLM router had no active backend connections for 10 minutes"
                                )
                                if not requires_image_inputs
                                else (
                                    f"Dashboard LLM router had no active image-capable backend connections for model {requested_model!r} for 10 minutes"
                                    if requested_model is not None
                                    else "Dashboard LLM router had no active image-capable backend connections for 10 minutes"
                                )
                            )
                        try:
                            await asyncio.wait_for(
                                self._condition.wait(),
                                timeout=remaining_seconds,
                            )
                        except asyncio.TimeoutError as exc:
                            raise NoActiveLLMConnectionError(
                                (
                                    f"Dashboard LLM router had no active backend connections for model {requested_model!r} for 10 minutes"
                                    if requested_model is not None
                                    else "Dashboard LLM router had no active backend connections for 10 minutes"
                                )
                                if not requires_image_inputs
                                else (
                                    f"Dashboard LLM router had no active image-capable backend connections for model {requested_model!r} for 10 minutes"
                                    if requested_model is not None
                                    else "Dashboard LLM router had no active image-capable backend connections for 10 minutes"
                                )
                            ) from exc
                finally:
                    self._waiting_requests -= 1

    async def _release_runtime(self, runtime: _ConnectionRuntime) -> None:
        async with self._condition:
            now = self._time_fn()
            runtime.in_flight_requests = max(0, runtime.in_flight_requests - 1)
            if runtime.in_flight_requests == 0 and runtime.busy_window_started_at is not None:
                runtime.busy_seconds_total += max(0.0, now - runtime.busy_window_started_at)
                runtime.busy_window_started_at = None
            self._condition.notify_all()

    def _runtime_usage_percentage(self, runtime: _ConnectionRuntime, *, now: float) -> float | None:
        observed_seconds = max(0.0, now - runtime.observed_since)
        if observed_seconds <= 0.0:
            return None
        busy_seconds = runtime.busy_seconds_total
        if runtime.busy_window_started_at is not None:
            busy_seconds += max(0.0, now - runtime.busy_window_started_at)
        return max(0.0, min(100.0, (busy_seconds / observed_seconds) * 100.0))

    def _available_runtimes(
        self,
        *,
        requires_image_inputs: bool = False,
        requested_model: str | None = None,
        required_total_tokens: int | None = None,
    ) -> list[_ConnectionRuntime]:
        base_candidates = [
            runtime
            for runtime in self._connections
            if runtime.active
            and (runtime.config.supports_image_inputs or not requires_image_inputs)
            and runtime.in_flight_requests < runtime.config.max_concurrent_requests
        ]
        if requested_model is None:
            candidates = base_candidates
        else:
            exact_candidates = [
                runtime for runtime in base_candidates if runtime.config.model == requested_model
            ]
            if exact_candidates:
                candidates = exact_candidates
            else:
                canonical_requested = normalize_canonical_model_name(requested_model)
                candidates = [
                    runtime
                    for runtime in base_candidates
                    if runtime.canonical_model == canonical_requested
                ]
        if requested_model is not None:
            return self._filter_runtimes_by_context(
                candidates,
                required_total_tokens=required_total_tokens,
            )
        candidates = self._filter_runtimes_by_context(
            candidates,
            required_total_tokens=required_total_tokens,
        )
        # For no-model requests, prefer least busy; tie-break by best recent tok/s
        return sorted(
            candidates,
            key=lambda item: (
                item.in_flight_requests / item.config.max_concurrent_requests,
                item.in_flight_requests,
                -(item.telemetry.average_output_tokens_per_second or 0.0),
                item.config.connection_id,
            ),
        )

    def _active_runtime_count(
        self,
        *,
        requires_image_inputs: bool = False,
        requested_model: str | None = None,
        required_total_tokens: int | None = None,
    ) -> int:
        return sum(
            1
            for runtime in self._connections
            if runtime.active
            and (runtime.config.supports_image_inputs or not requires_image_inputs)
            and (requested_model is None or runtime.config.model == requested_model)
            and self._runtime_meets_context_requirement(
                runtime,
                required_total_tokens=required_total_tokens,
            )
        )

    def _has_alternative_active_runtime(
        self,
        excluded_connection_ids: set[str],
        *,
        requires_image_inputs: bool = False,
        requested_model: str | None = None,
        required_total_tokens: int | None = None,
    ) -> bool:
        return any(
            runtime.active
            and runtime.config.connection_id not in excluded_connection_ids
            and (runtime.config.supports_image_inputs or not requires_image_inputs)
            and (requested_model is None or runtime.config.model == requested_model)
            and self._runtime_meets_context_requirement(
                runtime,
                required_total_tokens=required_total_tokens,
            )
            for runtime in self._connections
        )

    @staticmethod
    def _extract_completion_budget_tokens(payload: dict[str, Any]) -> int:
        for key in ("max_completion_tokens", "max_tokens"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                return max(GLOBAL_MIN_COMPLETION_TOKENS, int(value))
            except (TypeError, ValueError):
                continue
        return GLOBAL_MIN_COMPLETION_TOKENS

    @classmethod
    def _estimate_prompt_tokens(cls, payload: dict[str, Any]) -> int:
        try:
            prompt_like: dict[str, Any] = {}
            for key in (
                "messages",
                "tools",
                "functions",
                "function_call",
                "response_format",
                "input",
            ):
                if key in payload:
                    prompt_like[key] = payload[key]
            serialized = str(prompt_like)
            estimated = int(math.ceil(len(serialized) / 4.0))
            # Heuristic image overhead for VLMs.
            image_items = 0
            messages = payload.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    content = message.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if not isinstance(item, dict):
                                continue
                            item_type = str(item.get("type") or "").strip().lower()
                            if item_type in {"image_url", "input_image", "image"}:
                                image_items += 1
            estimated += image_items * 2048
            return max(1, estimated)
        except Exception:
            return GLOBAL_MIN_COMPLETION_TOKENS

    @classmethod
    def _estimate_required_total_tokens(cls, payload: dict[str, Any]) -> int:
        completion_budget = cls._extract_completion_budget_tokens(payload)
        prompt_tokens = cls._estimate_prompt_tokens(payload)
        return max(1, prompt_tokens + completion_budget)

    @staticmethod
    def _runtime_meets_context_requirement(
        runtime: _ConnectionRuntime,
        *,
        required_total_tokens: int | None,
    ) -> bool:
        if required_total_tokens is None:
            return True
        if runtime.max_context_tokens is None:
            return True
        return runtime.max_context_tokens >= required_total_tokens

    @classmethod
    def _filter_runtimes_by_context(
        cls,
        runtimes: list[_ConnectionRuntime],
        *,
        required_total_tokens: int | None,
    ) -> list[_ConnectionRuntime]:
        if required_total_tokens is None:
            return list(runtimes)
        known = [runtime for runtime in runtimes if runtime.max_context_tokens is not None]
        if not known:
            return list(runtimes)
        fitting = [
            runtime
            for runtime in known
            if runtime.max_context_tokens is not None and runtime.max_context_tokens >= required_total_tokens
        ]
        return fitting

    def _has_any_capacity_candidate(
        self,
        *,
        requires_image_inputs: bool,
        requested_model: str | None,
        required_total_tokens: int | None,
    ) -> bool:
        candidates = [
            runtime
            for runtime in self._connections
            if runtime.active
            and (runtime.config.supports_image_inputs or not requires_image_inputs)
            and (requested_model is None or runtime.config.model == requested_model or runtime.canonical_model == normalize_canonical_model_name(requested_model))
        ]
        return bool(
            self._filter_runtimes_by_context(
                candidates,
                required_total_tokens=required_total_tokens,
            )
        )

    def _max_known_context_tokens(
        self,
        *,
        requires_image_inputs: bool,
        requested_model: str | None,
    ) -> int | None:
        known_values = [
            runtime.max_context_tokens
            for runtime in self._connections
            if runtime.active
            and runtime.max_context_tokens is not None
            and (runtime.config.supports_image_inputs or not requires_image_inputs)
            and (requested_model is None or runtime.config.model == requested_model or runtime.canonical_model == normalize_canonical_model_name(requested_model))
        ]
        if not known_values:
            return None
        return max(int(value) for value in known_values if value is not None)

    @classmethod
    def _infer_runtime_context_tokens(
        cls,
        runtime: _ConnectionRuntime,
        models_body: object,
    ) -> int | None:
        records = cls._models_probe_records(models_body)
        if not records:
            return runtime.max_context_tokens
        exact: list[int] = []
        fallback: list[int] = []
        runtime_model = str(runtime.config.model or "")
        runtime_canonical = normalize_canonical_model_name(runtime_model)
        for record in records:
            context_tokens = cls._extract_context_tokens_from_model_record(record)
            if context_tokens is None:
                continue
            fallback.append(context_tokens)
            names = cls._model_record_names(record)
            if runtime_model in names:
                exact.append(context_tokens)
                continue
            canonical_names = [normalize_canonical_model_name(name) for name in names]
            if runtime_canonical and runtime_canonical in canonical_names:
                exact.append(context_tokens)
        if exact:
            return max(exact)
        if fallback:
            return max(fallback)
        return runtime.max_context_tokens

    @staticmethod
    def _models_probe_records(models_body: object) -> list[dict[str, Any]]:
        if isinstance(models_body, list):
            return [item for item in models_body if isinstance(item, dict)]
        if isinstance(models_body, dict):
            data = models_body.get("data")
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            models = models_body.get("models")
            if isinstance(models, list):
                return [item for item in models if isinstance(item, dict)]
        return []

    @staticmethod
    def _model_record_names(record: dict[str, Any]) -> list[str]:
        names: list[str] = []
        for key in ("id", "name", "model"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
        return names

    @classmethod
    def _extract_context_tokens_from_model_record(
        cls,
        record: dict[str, Any],
    ) -> int | None:
        candidate_keys = (
            "context_length",
            "max_context_length",
            "max_model_len",
            "max_sequence_length",
            "max_seq_len",
            "max_position_embeddings",
            "n_ctx",
            "num_ctx",
            "context_window",
        )
        search_spaces: list[dict[str, Any]] = [record]
        for nested_key in ("metadata", "details", "model_info", "capabilities"):
            nested = record.get(nested_key)
            if isinstance(nested, dict):
                search_spaces.append(nested)
        extracted: list[int] = []
        for space in search_spaces:
            for key in candidate_keys:
                value = space.get(key)
                parsed = cls._parse_positive_int(value)
                if parsed is not None:
                    extracted.append(parsed)
        if not extracted:
            return None
        return max(extracted)

    @staticmethod
    def _parse_positive_int(value: Any) -> int | None:
        try:
            parsed = int(str(value).strip())
        except Exception:
            return None
        if parsed <= 0:
            return None
        return parsed

    @staticmethod
    def _payload_requires_image_inputs(payload: dict[str, Any]) -> bool:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return False
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type") or "").strip().lower()
                    if item_type in {"image_url", "input_image", "image"}:
                        return True
        return False

    def _runtime_for_id(self, connection_id: str) -> _ConnectionRuntime:
        for runtime in self._connections:
            if runtime.config.connection_id == connection_id:
                return runtime
        raise KeyError(connection_id)

    @staticmethod
    def _can_reuse_runtime(
        current: LLMRouterConnectionConfig,
        replacement: LLMRouterConnectionConfig,
    ) -> bool:
        return (
            current.connection_id == replacement.connection_id
            and current.base_url == replacement.base_url
            and current.api_key == replacement.api_key
            and current.transport_mode == replacement.transport_mode
        )

    @staticmethod
    def _relative_connection_path(
        config: LLMRouterConnectionConfig,
        suffix: str,
    ) -> str:
        if DashboardLLMRouter._uses_pull_worker_transport(config):
            raise ValueError("Pull-request worker connections do not expose direct HTTP paths")
        base_path = urlparse(config.base_url).path.rstrip("/")
        normalized_suffix = suffix.lstrip("/")
        if base_path.endswith("/v1"):
            return normalized_suffix
        return f"v1/{normalized_suffix}"

    @staticmethod
    def _normalize_chat_payload_for_connection(
        payload: dict[str, Any],
        config: LLMRouterConnectionConfig,
    ) -> dict[str, Any]:
        normalized = dict(payload)
        hostname = (urlparse(config.base_url).hostname or "").lower()
        model_name = (config.model or "").lower()
        if "max_completion_tokens" in normalized:
            try:
                normalized["max_completion_tokens"] = max(
                    GLOBAL_MIN_COMPLETION_TOKENS,
                    int(normalized["max_completion_tokens"]),
                )
            except (TypeError, ValueError):
                normalized["max_completion_tokens"] = GLOBAL_MIN_COMPLETION_TOKENS
        elif "max_tokens" in normalized:
            try:
                normalized["max_tokens"] = max(
                    GLOBAL_MIN_COMPLETION_TOKENS,
                    int(normalized["max_tokens"]),
                )
            except (TypeError, ValueError):
                normalized["max_tokens"] = GLOBAL_MIN_COMPLETION_TOKENS
        else:
            normalized["max_tokens"] = GLOBAL_MIN_COMPLETION_TOKENS
        if (
            hostname == "api.openai.com"
            and model_name.startswith("gpt-5")
            and "max_tokens" in normalized
            and "max_completion_tokens" not in normalized
        ):
            normalized["max_completion_tokens"] = normalized.pop("max_tokens")
        if hostname == "api.openai.com" and model_name.startswith("gpt-5"):
            normalized.pop("temperature", None)
        if hostname in {"api.openai.com", "api.together.xyz"}:
            normalized.pop("chat_template_kwargs", None)
        if "qwen" in model_name:
            normalized["temperature"] = 1.0
            normalized["top_p"] = 0.95
            normalized["presence_penalty"] = 1.5
            extra_body = normalized.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
            else:
                extra_body = dict(extra_body)
            extra_body["top_k"] = 20
            normalized["extra_body"] = extra_body
        return normalized

    @staticmethod
    def _extract_usage(usage: Any) -> tuple[int, int]:
        if not isinstance(usage, dict):
            raise MissingUsageTelemetryError(
                "Upstream chat-completions response did not include usage telemetry"
            )
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
            raise MissingUsageTelemetryError(
                "Upstream chat-completions response usage must include integer prompt_tokens and completion_tokens"
            )
        return prompt_tokens, completion_tokens

    @staticmethod
    def _is_retryable_upstream_status(exc: httpx.HTTPStatusError) -> bool:
        status_code = exc.response.status_code
        return status_code in {429, 500, 502, 503, 504}

    @staticmethod
    def _should_deactivate_runtime_after_status(exc: httpx.HTTPStatusError) -> bool:
        return exc.response.status_code in {401, 403, 404}

    @staticmethod
    def _format_upstream_error(exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            message = str(exc)
            try:
                body = exc.response.text.strip()
            except Exception:  # noqa: BLE001
                body = ""
            if body:
                condensed = " ".join(body.split())
                return f"{message} | upstream body: {condensed[:300]}"
            return message
        return str(exc)

    @staticmethod
    def _uses_pull_worker_transport(config: LLMRouterConnectionConfig) -> bool:
        return config.source_kind == "worker" and config.transport_mode == "pull-request"

    def _transient_backoff_for_attempt(self, attempt_index: int) -> float:
        if self._transient_failure_backoff_seconds <= 0.0:
            return 0.0
        return self._transient_failure_backoff_seconds * (attempt_index + 1)
