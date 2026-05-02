from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
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


def _router_context(
    *,
    panel: dict[str, Any],
    history: dict[str, Any],
    requests: dict[str, Any],
) -> dict[str, Any]:
    normalized_requests = _normalize_request_entries(panel=panel, requests=requests)
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
    templates.env.cache = None
    templates.env.globals["_format_compact_metric"] = _format_compact_metric

    app = FastAPI(title="LLM Router UI")
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
    app.mount("/static", StaticFiles(directory=str(root / "static")), name="static")

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        try:
            await _fetch_backend_json("/api/router/panel")
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"ok": False, "detail": str(exc)}, status_code=503)
        return JSONResponse({"ok": True})

    @app.get("/", response_class=HTMLResponse)
    async def router_index(request: Request) -> HTMLResponse:
        error = request.query_params.get("router_error")
        try:
            panel = await _fetch_backend_json("/api/router/panel")
            history = await _fetch_backend_json("/api/router/history")
            requests = await _fetch_backend_json_optional(
                "/api/router/requests",
                default={"requests": [], "total_retained": 0, "max_entries": 0},
            )
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
                    requests={"requests": [], "total_retained": 0, "max_entries": 0},
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
            panel = await _fetch_backend_json("/api/router/panel")
            history = await _fetch_backend_json("/api/router/history")
            requests = await _fetch_backend_json_optional(
                "/api/router/requests",
                default={"requests": [], "total_retained": 0, "max_entries": 0},
            )
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
