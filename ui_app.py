from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BACKEND_BASE_URL = os.getenv("LLM_ROUTER_BACKEND_URL", "http://127.0.0.1:8788").rstrip("/")
ROUTER_DISPLAY_TIMEZONE = ZoneInfo("Europe/London")
REQUEST_LOG_THROUGHPUT_WINDOW = timedelta(minutes=10)
UI_REQUEST_LOG_LIMIT = 2000
UI_BACKEND_CACHE_SECONDS = 2.0
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _external_router_base_url(request: Request) -> str:
    hostname = request.url.hostname or "127.0.0.1"
    scheme = request.url.scheme or "http"
    return f"{scheme}://{hostname}:8788/v1"


def _format_compact_metric(value: int) -> str:
    abs_value = abs(int(value))
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _copy_proxy_headers(response: httpx.Response) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for name, value in response.headers.items():
        if name.lower() in HOP_BY_HOP_HEADERS:
            continue
        forwarded[name] = value
    return forwarded


def _request_timestamp_label(value: str | None) -> str:
    if not value:
        return "Unknown time"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone(ROUTER_DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_request_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _request_log_window_metrics(
    requests: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    entries = requests.get("requests")
    if not isinstance(entries, list):
        entries = []
    window_end = now.astimezone(timezone.utc) if now is not None else _utc_now()
    cutoff = window_end - REQUEST_LOG_THROUGHPUT_WINDOW
    metrics_samples = _request_log_metric_samples(entries=entries, cutoff=cutoff, window_end=window_end)
    duration_samples = [item["duration_seconds"] for item in metrics_samples]
    total_completion_tokens = sum(item["completion_tokens"] for item in metrics_samples)
    total_duration_seconds = sum(duration_samples)
    window_seconds = max(REQUEST_LOG_THROUGHPUT_WINDOW.total_seconds(), 1.0)
    average_throughput = total_completion_tokens / window_seconds
    average_request_seconds = (
        sum(duration_samples) / len(duration_samples)
        if duration_samples
        else 0.0
    )
    return {
        "request_log_output_tokens_per_second": round(average_throughput, 2),
        "request_log_average_request_seconds": round(average_request_seconds, 2),
        "request_log_sample_count": len(metrics_samples),
        "request_log_window_minutes": int(REQUEST_LOG_THROUGHPUT_WINDOW.total_seconds() // 60),
    }


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


def _request_log_throughput_by_connection(
    requests: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, float]:
    entries = requests.get("requests")
    if not isinstance(entries, list):
        return {}
    window_end = now.astimezone(timezone.utc) if now is not None else _utc_now()
    cutoff = window_end - REQUEST_LOG_THROUGHPUT_WINDOW
    totals_by_connection: dict[str, float] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        timestamp = _parse_request_timestamp(str(item.get("timestamp") or "").strip())
        if timestamp is None or timestamp < cutoff or timestamp > window_end:
            continue
        connection = item.get("connection")
        connection_id = ""
        if isinstance(connection, dict):
            connection_id = str(connection.get("connection_id") or "").strip()
            if not connection_id:
                nested_worker_id = str(connection.get("worker_id") or "").strip()
                nested_source_kind = str(connection.get("source_kind") or "").strip().lower()
                if nested_source_kind == "worker" and nested_worker_id:
                    connection_id = f"worker:{nested_worker_id}"
        if not connection_id:
            worker_id = str(item.get("worker_id") or "").strip()
            source_kind = str(item.get("source_kind") or "").strip().lower()
            selected_model = str(item.get("selected_model") or item.get("model") or "").strip()
            base_url = str(item.get("base_url") or "").strip()
            if source_kind == "worker" and worker_id:
                connection_id = f"worker:{worker_id}"
            elif base_url and selected_model:
                connection_id = _router_connection_id(
                    explicit_value=None,
                    base_url=base_url,
                    model=selected_model,
                    ordinal=1,
                )
        if not connection_id:
            continue
        completion_tokens = item.get("completion_tokens")
        duration_seconds = item.get("duration_seconds")
        if not isinstance(completion_tokens, (int, float)):
            continue
        if not isinstance(duration_seconds, (int, float)) or float(duration_seconds) <= 0.0:
            continue
        totals_by_connection[connection_id] = totals_by_connection.get(connection_id, 0.0) + float(
            completion_tokens
        )
    window_seconds = max(REQUEST_LOG_THROUGHPUT_WINDOW.total_seconds(), 1.0)
    return {
        connection_id: round(total_completion_tokens / window_seconds, 2)
        for connection_id, total_completion_tokens in totals_by_connection.items()
    }


def _connection_with_request_log_throughput(
    connection: dict[str, Any],
    *,
    throughput_by_connection: dict[str, float],
) -> dict[str, Any]:
    next_connection = dict(connection)
    telemetry = next_connection.get("telemetry")
    if isinstance(telemetry, dict):
        next_telemetry = dict(telemetry)
    else:
        next_telemetry = {}
    connection_id = str(next_connection.get("connection_id") or "").strip()
    next_telemetry["average_output_tokens_per_second"] = throughput_by_connection.get(connection_id, 0.0)
    next_connection["telemetry"] = next_telemetry
    return next_connection


def _panel_with_request_log_connection_metrics(
    panel: dict[str, Any],
    *,
    requests: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    throughput_by_connection = _request_log_throughput_by_connection(requests, now=now)
    next_panel = dict(panel)

    connections = panel.get("connections")
    if isinstance(connections, list):
        next_panel["connections"] = [
            _connection_with_request_log_throughput(item, throughput_by_connection=throughput_by_connection)
            if isinstance(item, dict)
            else item
            for item in connections
        ]

    worker_connections = panel.get("worker_connections")
    if isinstance(worker_connections, list):
        next_panel["worker_connections"] = [
            _connection_with_request_log_throughput(item, throughput_by_connection=throughput_by_connection)
            if isinstance(item, dict)
            else item
            for item in worker_connections
        ]

    endpoints = panel.get("endpoints")
    if isinstance(endpoints, list):
        next_endpoints: list[dict[str, Any]] = []
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                continue
            next_endpoint = dict(endpoint)
            models = endpoint.get("models")
            if isinstance(models, list):
                next_models: list[dict[str, Any]] = []
                endpoint_total_tps = 0.0
                for model in models:
                    if not isinstance(model, dict):
                        continue
                    next_model = _connection_with_request_log_throughput(
                        model,
                        throughput_by_connection=throughput_by_connection,
                    )
                    model_tps = next_model.get("telemetry", {}).get("average_output_tokens_per_second")
                    if isinstance(model_tps, (int, float)):
                        endpoint_total_tps += float(model_tps)
                    next_models.append(next_model)
                next_endpoint["models"] = next_models
                next_endpoint["average_output_tokens_per_second"] = round(endpoint_total_tps, 2)
            next_endpoints.append(next_endpoint)
        next_panel["endpoints"] = next_endpoints

    return next_panel


def _request_log_metric_samples(
    *,
    entries: list[dict[str, Any]],
    cutoff: datetime,
    window_end: datetime,
) -> list[dict[str, float]]:
    metric_samples: list[dict[str, float]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        timestamp = _parse_request_timestamp(str(item.get("timestamp") or "").strip())
        if timestamp is None or timestamp < cutoff or timestamp > window_end:
            continue
        completion_tokens = item.get("completion_tokens")
        duration_seconds = item.get("duration_seconds")
        if not isinstance(completion_tokens, (int, float)):
            continue
        if not isinstance(duration_seconds, (int, float)):
            continue
        resolved_duration = float(duration_seconds)
        if resolved_duration <= 0.0:
            continue
        metric_samples.append(
            {
                "duration_seconds": resolved_duration,
                "completion_tokens": float(completion_tokens),
                "throughput": float(completion_tokens) / resolved_duration,
            }
        )
    return metric_samples


def _history_with_request_log_throughput(
    *,
    history: dict[str, Any],
    requests: dict[str, Any],
) -> dict[str, Any]:
    points = history.get("points")
    entries = requests.get("requests")
    if not isinstance(points, list) or not isinstance(entries, list):
        return history
    next_history = dict(history)
    next_points: list[dict[str, Any]] = []
    for item in points:
        if not isinstance(item, dict):
            continue
        point = dict(item)
        timestamp = _parse_request_timestamp(str(point.get("timestamp") or "").strip())
        if timestamp is None:
            next_points.append(point)
            continue
        cutoff = timestamp - REQUEST_LOG_THROUGHPUT_WINDOW
        throughput_samples = [
            item
            for item in _request_log_metric_samples(
            entries=entries,
            cutoff=cutoff,
            window_end=timestamp,
        )
        ]
        total_completion_tokens = sum(item["completion_tokens"] for item in throughput_samples)
        window_seconds = max(REQUEST_LOG_THROUGHPUT_WINDOW.total_seconds(), 1.0)
        point["throughput"] = round(
            total_completion_tokens / window_seconds,
            3,
        ) if throughput_samples else 0.0
        next_points.append(point)
    next_history["points"] = next_points
    return next_history


def _normalize_request_entries(
    *,
    panel: dict[str, Any],
    requests: dict[str, Any],
) -> dict[str, Any]:
    worker_names: dict[str, str] = {}
    workers = panel.get("workers")
    if isinstance(workers, list):
        for item in workers:
            if not isinstance(item, dict):
                continue
            worker_id = str(item.get("worker_id") or "").strip()
            display_name = str(item.get("display_name") or "").strip()
            if worker_id and display_name:
                worker_names[worker_id] = display_name
    normalized = dict(requests)
    entries = requests.get("requests")
    if not isinstance(entries, list):
        normalized["requests"] = []
        return normalized
    next_entries: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        timestamp = str(entry.get("timestamp") or "").strip()
        entry["timestamp_label"] = _request_timestamp_label(timestamp)
        source_kind = str(entry.get("source_kind") or "").strip()
        worker_id = str(entry.get("worker_id") or "").strip()
        base_url = str(entry.get("base_url") or "").strip()
        if source_kind == "worker" and worker_id:
            entry["connection_label"] = worker_names.get(worker_id, worker_id)
        elif base_url:
            entry["connection_label"] = base_url
        else:
            entry["connection_label"] = "Unknown route"
        next_entries.append(entry)
    normalized["requests"] = next_entries
    return normalized


def _iter_text_fragments(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    fragments: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                fragments.append(text)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("text", "content", "value", "input_text", "output_text"):
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate.strip():
                fragments.append(candidate.strip())
                break
    return fragments


def _extract_request_input_blocks(entry: dict[str, Any]) -> list[dict[str, str]]:
    request_payload = entry.get("request")
    if not isinstance(request_payload, dict):
        return []
    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        return []
    blocks: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        text = "\n\n".join(_iter_text_fragments(item.get("content")))
        if not text:
            continue
        blocks.append(
            {
                "role": str(item.get("role") or "message").strip() or "message",
                "text": text,
            }
        )
    return blocks


def _extract_request_image_blocks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    request_payload = entry.get("request")
    if not isinstance(request_payload, dict):
        return []
    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        return []
    blocks: list[dict[str, Any]] = []
    image_index = 0
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "message").strip() or "message"
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            image = _extract_image_src_from_part(part)
            if image is None:
                continue
            image_index += 1
            blocks.append(
                {
                    "role": role,
                    "index": image_index,
                    "source": image,
                    "is_data_url": image.startswith("data:image/"),
                }
            )
    return blocks


def _extract_image_src_from_part(part: dict[str, Any]) -> str | None:
    part_type = str(part.get("type") or "").strip().lower()
    if part_type not in {"image", "image_url", "input_image"} and not any(
        key in part for key in ("image_url", "input_image", "image")
    ):
        return None
    candidates = [
        part.get("image_url"),
        part.get("input_image"),
        part.get("image"),
    ]
    for candidate in candidates:
        extracted = _extract_image_src(candidate)
        if extracted is not None:
            return extracted
    return None


def _extract_image_src(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
        return None
    if isinstance(value, dict):
        for key in ("url", "image_url", "data"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _extract_response_text_blocks(entry: dict[str, Any]) -> tuple[list[str], list[str]]:
    response_payload = entry.get("response")
    if not isinstance(response_payload, dict):
        return [], []
    reasoning_parts: list[str] = []
    output_parts: list[str] = []
    choices = response_payload.get("choices")
    if not isinstance(choices, list):
        return reasoning_parts, output_parts
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        reasoning_parts.extend(_iter_text_fragments(message.get("reasoning")))
        for key in ("reasoning_content", "reasoning_text"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                reasoning_parts.append(value.strip())
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    part_type = str(part.get("type") or "").lower()
                    text_values = _iter_text_fragments([part])
                    if not text_values:
                        continue
                    if "reason" in part_type or "think" in part_type:
                        reasoning_parts.extend(text_values)
                    else:
                        output_parts.extend(text_values)
                elif isinstance(part, str) and part.strip():
                    output_parts.append(part.strip())
        else:
            output_parts.extend(_iter_text_fragments(content))
    return reasoning_parts, output_parts


async def _fetch_backend_json(path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{BACKEND_BASE_URL}{path}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail=f"Malformed backend payload for {path}")
        return payload


async def _fetch_backend_json_optional(path: str, *, default: dict[str, Any]) -> dict[str, Any]:
    try:
        return await _fetch_backend_json(path)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return default
        raise


def _ui_request_log_default() -> dict[str, Any]:
    return {"requests": [], "total_retained": 0, "max_entries": 0}


async def _fetch_router_payloads(app: FastAPI) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cache_entry = getattr(app.state, "router_ui_backend_cache", None)
    now = time.monotonic()
    if isinstance(cache_entry, dict) and (now - float(cache_entry.get("captured_at") or 0.0)) < UI_BACKEND_CACHE_SECONDS:
        return (
            dict(cache_entry["panel"]),
            dict(cache_entry["history"]),
            dict(cache_entry["requests"]),
        )
    panel = await _fetch_backend_json("/api/router/panel")
    history = await _fetch_backend_json("/api/router/history")
    requests = await _fetch_backend_json_optional(
        f"/api/router/requests?limit={UI_REQUEST_LOG_LIMIT}",
        default=_ui_request_log_default(),
    )
    app.state.router_ui_backend_cache = {
        "captured_at": now,
        "panel": panel,
        "history": history,
        "requests": requests,
    }
    return dict(panel), dict(history), dict(requests)


def _router_context(
    *,
    panel: dict[str, Any],
    history: dict[str, Any],
    requests: dict[str, Any],
) -> dict[str, Any]:
    normalized_requests = _normalize_request_entries(panel=panel, requests=requests)
    panel = _panel_with_request_log_connection_metrics(panel, requests=normalized_requests)
    history_with_request_log_throughput = _history_with_request_log_throughput(
        history=history,
        requests=normalized_requests,
    )
    request_log_metrics = _request_log_window_metrics(normalized_requests)
    panel_with_request_metrics = dict(panel)
    panel_with_request_metrics.update(request_log_metrics)
    panel_with_request_metrics["total_output_tokens_per_second"] = request_log_metrics[
        "request_log_output_tokens_per_second"
    ]
    panel_with_request_metrics["average_output_tokens_per_second"] = request_log_metrics[
        "request_log_output_tokens_per_second"
    ]
    panel_with_request_metrics["average_request_seconds"] = request_log_metrics[
        "request_log_average_request_seconds"
    ]
    return {
        "router_panel": panel_with_request_metrics,
        "router_history": history_with_request_log_throughput,
        "router_requests": normalized_requests,
        "page_title": "LLM Router",
        "error_count": int(panel_with_request_metrics.get("error_count") or 0),
    }


def create_router_ui_app() -> FastAPI:
    root = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(root / "templates"))
    templates.env.globals["_format_compact_metric"] = _format_compact_metric

    app = FastAPI(title="LLM Router UI")
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
    app.mount("/static", StaticFiles(directory=str(root / "static")), name="static")
    app.state.router_ui_backend_cache = None

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        try:
            await _fetch_router_payloads(app)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"ok": False, "detail": str(exc)}, status_code=503)
        return JSONResponse({"ok": True})

    @app.get("/", response_class=HTMLResponse)
    async def router_index(request: Request) -> HTMLResponse:
        error = request.query_params.get("router_error")
        try:
            panel, history, requests = await _fetch_router_payloads(app)
        except Exception as exc:  # noqa: BLE001
            fallback_panel = {
                "base_url": BACKEND_BASE_URL,
                "shared_model": "",
                "public_access": {},
                "connections": [],
                "endpoints": [],
                "workers": [],
                "worker_connections": [],
                "worker_counts": {"online": 0, "offline": 0, "total": 0},
                "waiting_requests": 0,
                "modal_connections": [],
                "modal_model_options": [],
                "modal_gpu_options": [],
                "total_tokens_label": "0",
                "average_request_seconds": 0.0,
                "average_output_tokens_per_second": 0.0,
                "total_output_tokens_per_second": 0.0,
                "total_in_flight_requests": 0,
                "rendered_at": "",
                "error": error or f"Backend unavailable: {exc}",
                "error_count": 0,
                "canonical_models": [],
            }
            return templates.TemplateResponse(
                request,
                "index.html",
                _router_context(
                    panel=fallback_panel,
                    history={"points": [], "window_seconds": 3600},
                    requests=_ui_request_log_default(),
                ),
                status_code=503,
            )
        if error:
            panel["error"] = error
        panel["base_url"] = _external_router_base_url(request)
        return templates.TemplateResponse(
            request,
            "index.html",
            _router_context(panel=panel, history=history, requests=requests),
        )

    @app.get("/router/live")
    async def router_live(request: Request) -> JSONResponse:
        try:
            panel, history, requests = await _fetch_router_payloads(app)
        except Exception:
            return JSONResponse({"error": "Backend unavailable"}, status_code=503)
        panel["base_url"] = _external_router_base_url(request)
        context = _router_context(panel=panel, history=history, requests=requests)
        return JSONResponse(
            {
                "summary_html": templates.env.get_template("_router_summary.html").render(context),
                "public_access_html": templates.env.get_template("_router_public_access.html").render(context),
                "stats_html": templates.env.get_template("_router_stats.html").render(context),
                "workers_html": templates.env.get_template("_router_workers.html").render(context),
                "endpoints_html": templates.env.get_template("_router_live.html").render(context),
                "requests_html": templates.env.get_template("_router_requests_panel.html").render(context),
                "rendered_at": panel.get("rendered_at"),
                "metrics_history": context["router_history"],
            }
        )

    @app.get("/router/requests/{request_id}", response_class=HTMLResponse)
    async def router_request_detail(request: Request, request_id: str) -> HTMLResponse:
        try:
            entry = await _fetch_backend_json(f"/api/router/requests/{request_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Request entry not found") from exc
            raise
        summary = entry.get("summary") if isinstance(entry, dict) else {}
        input_blocks = _extract_request_input_blocks(entry)
        image_blocks = _extract_request_image_blocks(entry)
        reasoning_blocks, output_blocks = _extract_response_text_blocks(entry)
        return templates.TemplateResponse(
            request,
            "request_detail.html",
            {
                "entry": entry,
                "summary": summary if isinstance(summary, dict) else {},
                "input_blocks": input_blocks,
                "image_blocks": image_blocks,
                "reasoning_blocks": reasoning_blocks,
                "output_blocks": output_blocks,
                "page_title": f"Router Request {request_id}",
            },
        )

    @app.get("/router/errors", response_class=HTMLResponse)
    async def router_errors(request: Request) -> HTMLResponse:
        payload = await _fetch_backend_json("/api/router/errors")
        return templates.TemplateResponse(
            request,
            "errors.html",
            {
                "errors": payload.get("errors") or [],
                "page_title": "Router Errors",
            },
        )

    @app.get("/router/errors/{entry_id}", response_class=HTMLResponse)
    async def router_error_detail(request: Request, entry_id: int) -> HTMLResponse:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{BACKEND_BASE_URL}/api/router/errors/{entry_id}")
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Error entry not found")
        response.raise_for_status()
        entry = response.json()
        return templates.TemplateResponse(
            request,
            "error_detail.html",
            {
                "entry": entry,
                "page_title": f"Router Error #{entry_id}",
            },
        )

    @app.get("/router/snapshot")
    async def router_snapshot() -> Response:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            response = await client.get(f"{BACKEND_BASE_URL}/router/snapshot")
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            headers=_copy_proxy_headers(response),
        )

    @app.api_route("/router/{path:path}", methods=["POST"])
    async def router_action_proxy(request: Request, path: str) -> Response:
        target_url = f"{BACKEND_BASE_URL}/router/{path}"
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=False) as client:
            response = await client.request(
                request.method,
                target_url,
                params=request.query_params,
                content=await request.body(),
                headers={
                    key: value
                    for key, value in request.headers.items()
                    if key.lower() in {"content-type", "accept"}
                },
            )
        location = response.headers.get("location")
        if location is not None and response.status_code in {301, 302, 303, 307, 308}:
            return RedirectResponse(location, status_code=response.status_code)
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            headers=_copy_proxy_headers(response),
        )

    return app


app = create_router_ui_app()
