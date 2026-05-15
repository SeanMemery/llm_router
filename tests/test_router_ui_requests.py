from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from ui_app import (
    _history_with_request_log_throughput,
    _panel_with_request_log_connection_metrics,
    _request_log_window_metrics,
    create_router_ui_app,
)


def test_request_log_window_metrics_uses_total_tokens_over_total_duration() -> None:
    metrics = _request_log_window_metrics(
        {
            "requests": [
                {
                    "timestamp": "2026-05-01T11:58:00Z",
                    "completion_tokens": 120,
                    "duration_seconds": 3.0,
                },
                {
                    "timestamp": "2026-05-01T11:55:00Z",
                    "completion_tokens": 30,
                    "duration_seconds": 1.0,
                },
                {
                    "timestamp": "2026-05-01T11:49:59Z",
                    "completion_tokens": 900,
                    "duration_seconds": 3.0,
                },
            ]
        },
        now=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert metrics["request_log_output_tokens_per_second"] == 0.25
    assert metrics["request_log_average_request_seconds"] == 2.0
    assert metrics["request_log_sample_count"] == 2


def test_history_uses_request_log_throughput_for_each_point() -> None:
    history = _history_with_request_log_throughput(
        history={
            "points": [
                {
                    "timestamp": "2026-05-01T12:00:00Z",
                    "epoch_seconds": 1.0,
                    "in_flight_requests": 1,
                    "throughput": 999.0,
                },
                {
                    "timestamp": "2026-05-01T12:06:00Z",
                    "epoch_seconds": 2.0,
                    "in_flight_requests": 2,
                    "throughput": 999.0,
                },
            ]
        },
        requests={
            "requests": [
                {
                    "timestamp": "2026-05-01T11:58:00Z",
                    "completion_tokens": 120,
                    "duration_seconds": 3.0,
                },
                {
                    "timestamp": "2026-05-01T12:05:00Z",
                    "completion_tokens": 30,
                    "duration_seconds": 1.0,
                },
                {
                    "timestamp": "2026-05-01T11:45:00Z",
                    "completion_tokens": 500,
                    "duration_seconds": 2.0,
                },
            ]
        },
    )

    assert history["points"][0]["throughput"] == 0.2
    assert history["points"][1]["throughput"] == 0.25


def test_panel_request_log_metrics_override_connection_throughput() -> None:
    panel = _panel_with_request_log_connection_metrics(
        {
            "connections": [
                {
                    "connection_id": "http-worker-example-v1-selected-model",
                    "telemetry": {"average_output_tokens_per_second": None},
                },
                {
                    "connection_id": "worker:worker-1",
                    "telemetry": {"average_output_tokens_per_second": None},
                },
            ],
            "worker_connections": [
                {
                    "connection_id": "worker:worker-1",
                    "telemetry": {"average_output_tokens_per_second": None},
                }
            ],
            "endpoints": [
                {
                    "base_url": "http://worker.example/v1",
                    "average_output_tokens_per_second": None,
                    "models": [
                        {
                            "connection_id": "http-worker-example-v1-selected-model",
                            "model": "selected-model",
                            "telemetry": {"average_output_tokens_per_second": None},
                        }
                    ],
                }
            ],
        },
        requests={
            "requests": [
                {
                    "timestamp": "2026-05-01T11:58:00Z",
                    "base_url": "http://worker.example/v1",
                    "selected_model": "selected-model",
                    "completion_tokens": 120,
                    "duration_seconds": 2.0,
                    "source_kind": "manual",
                },
                {
                    "timestamp": "2026-05-01T11:57:00Z",
                    "worker_id": "worker-1",
                    "completion_tokens": 60,
                    "duration_seconds": 3.0,
                    "source_kind": "worker",
                },
            ]
        },
        now=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert panel["connections"][0]["telemetry"]["average_output_tokens_per_second"] == 0.2
    assert panel["connections"][1]["telemetry"]["average_output_tokens_per_second"] == 0.1
    assert panel["worker_connections"][0]["telemetry"]["average_output_tokens_per_second"] == 0.1
    assert panel["endpoints"][0]["average_output_tokens_per_second"] == 0.2
    assert panel["endpoints"][0]["models"][0]["telemetry"]["average_output_tokens_per_second"] == 0.2


def test_router_ui_home_includes_request_panel(monkeypatch) -> None:
    monkeypatch.setattr(
        "ui_app._utc_now",
        lambda: datetime(2026, 4, 27, 12, 5, 0, tzinfo=timezone.utc),
    )

    async def _fake_fetch(path: str) -> dict[str, object]:
        if path == "/api/router/panel":
            return {
                "base_url": "http://router.example:8788/v1",
                "shared_model": "shared-model",
                "public_access": {},
                "connections": [],
                "endpoints": [],
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "display_name": "s1804274-infk8s-40gb-router-worker-bjbwj-wnpwf",
                    }
                ],
                "worker_connections": [
                    {
                        "connection_id": "worker:worker-1",
                        "worker_id": "worker-1",
                        "transport_mode": "pull-request",
                        "telemetry": {
                            "request_count": 0,
                            "total_prompt_tokens": 0,
                            "total_completion_tokens": 0,
                            "average_output_tokens_per_second": None,
                            "average_recent_request_seconds": None,
                        },
                        "max_concurrent_requests": 1,
                        "in_flight_requests": 0,
                        "canonical_model": "selected-model",
                        "usage_percentage": 0.0,
                        "active": True,
                        "manually_disabled": False,
                    }
                ],
                "worker_counts": {"online": 0, "offline": 0, "total": 0},
                "waiting_requests": 0,
                "total_tokens_label": "4",
                "average_request_seconds": 0.2,
                "average_output_tokens_per_second": 20.0,
                "total_output_tokens_per_second": 20.0,
                "total_in_flight_requests": 0,
                "rendered_at": "12:00:00",
                "error_count": 0,
                "canonical_models": [],
            }
        if path == "/api/router/history":
            return {"points": [], "window_seconds": 3600}
        if path == "/api/router/requests":
            return {
                "max_entries": 1000,
                "total_retained": 1,
                "requests": [
                    {
                        "id": "req-1",
                        "timestamp": "2026-04-27T12:00:00Z",
                        "timestamp_label": "2026-04-27 12:00:00 UTC",
                        "status": "completed",
                        "status_label": "Completed",
                        "requested_model": "requested-model",
                        "selected_model": "selected-model",
                        "base_url": "worker://worker-1",
                        "base_url_display": "worker://worker-1",
                        "prompt_tokens": 10,
                        "completion_tokens": 1200,
                        "prompt_tokens_label": "10",
                        "completion_tokens_label": "1.20K",
                        "duration_seconds": 0.4,
                        "duration_label": "400ms",
                        "http_status_code": 200,
                        "source_kind": "worker",
                        "transport_mode": "pull-request",
                        "worker_id": "worker-1",
                    }
                ],
            }
        if path == "/api/router/requests/req-1":
            return {
                "id": "req-1",
                "status": "completed",
                "http_status_code": 200,
                "requested_model": "requested-model",
                "request_path": "/v1/chat/completions",
                "received_via_public_router": False,
                "request_headers": {"content-type": "application/json"},
                "response_headers": {"X-LLM-Model": "selected-model"},
                "request": {
                    "model": "requested-model",
                    "messages": [
                        {"role": "system", "content": "Be concise."},
                        {"role": "user", "content": "Explain the route choice."},
                    ],
                },
                "response": {
                    "id": "chatcmpl-1",
                    "choices": [
                        {
                            "message": {
                                "reasoning": "The router picked the lowest-cost healthy backend.",
                                "content": "It selected the healthy worker with available capacity.",
                            }
                        }
                    ],
                },
                "error": None,
                "usage": {"prompt_tokens": 10, "completion_tokens": 1200, "total_tokens": 1210},
                "timing": {
                    "router_request_seconds": 0.4,
                    "upstream_request_seconds": 0.4,
                    "end_to_end_request_seconds": 0.5,
                    "queue_wait_seconds": 0.1,
                },
                "connection": {
                    "connection_id": "conn-1",
                    "base_url": "worker://worker-1",
                    "selected_model": "selected-model",
                    "source_kind": "worker",
                    "transport_mode": "pull-request",
                    "worker_id": "worker-1",
                    "worker_request_id": "worker-request-1",
                },
                "summary": {
                    "id": "req-1",
                    "timestamp_label": "2026-04-27 12:00:00 UTC",
                    "status": "completed",
                    "status_label": "Completed",
                    "prompt_tokens_label": "10",
                    "completion_tokens_label": "20",
                    "duration_label": "400ms",
                    "http_status_code": 200,
                    "requested_model": "requested-model",
                    "selected_model": "selected-model",
                    "base_url": "http://worker.example/v1",
                },
            }
        raise AssertionError(f"Unexpected backend path: {path}")

    monkeypatch.setattr("ui_app._fetch_backend_json", _fake_fetch)
    app = create_router_ui_app()
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "Request Log" in home.text
        assert "Settings" in home.text
        assert "Stats" in home.text
        assert "Throughput" in home.text
        assert "(10m)" in home.text
        assert "0.4 s" in home.text
        assert "2 t/s" in home.text
        assert 'data-endpoint-support-toggle="modal"' in home.text


def test_router_ui_shows_inactive_automatic_worker(monkeypatch) -> None:
    async def _fake_fetch(path: str) -> dict[str, object]:
        if path == "/api/router/panel":
            return {
                "base_url": "http://router.example:8788/v1",
                "shared_model": "shared-model",
                "public_access": {},
                "connections": [],
                "endpoints": [],
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "display_name": "auto-worker-1",
                        "model": "demo-model",
                        "transport_mode": "pull-request",
                        "metadata": {},
                    }
                ],
                "worker_connections": [
                    {
                        "connection_id": "conn-1",
                        "worker_id": "worker-1",
                        "base_url": "worker://worker-1",
                        "model": "demo-model",
                        "source_kind": "worker",
                        "transport_mode": "pull-request",
                        "active": False,
                        "manually_disabled": True,
                        "in_flight_requests": 0,
                        "max_concurrent_requests": 2,
                        "usage_percentage": None,
                        "telemetry": {
                            "request_count": 0,
                            "total_prompt_tokens": 0,
                            "total_completion_tokens": 0,
                            "average_recent_request_seconds": None,
                            "average_output_tokens_per_second": None,
                        },
                    }
                ],
                "worker_counts": {"online": 1, "offline": 0, "total": 1},
                "waiting_requests": 0,
                "total_tokens_label": "0",
                "average_request_seconds": 0.0,
                "average_output_tokens_per_second": 0.0,
                "total_output_tokens_per_second": 0.0,
                "total_in_flight_requests": 0,
                "rendered_at": "12:00:00",
                "error_count": 0,
                "canonical_models": [],
            }
        if path == "/api/router/history":
            return {"points": [], "window_seconds": 3600}
        if path == "/api/router/requests":
            return {"max_entries": 1000, "total_retained": 0, "requests": []}
        raise AssertionError(f"Unexpected backend path: {path}")

    monkeypatch.setattr("ui_app._fetch_backend_json", _fake_fetch)
    monkeypatch.setattr(
        "ui_app._fetch_backend_json_optional",
        lambda path, default: _fake_fetch(path),
    )
    app = create_router_ui_app()
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "Automatic</p>" in home.text
        assert "auto-worker-1" in home.text
        assert "Inactive" in home.text


def test_router_live_returns_request_log_throughput_history(monkeypatch) -> None:
    monkeypatch.setattr(
        "ui_app._utc_now",
        lambda: datetime(2026, 5, 1, 12, 6, 0, tzinfo=timezone.utc),
    )

    async def _fake_fetch(path: str) -> dict[str, object]:
        if path == "/api/router/panel":
            return {
                "base_url": "http://router.example:8788/v1",
                "shared_model": "shared-model",
                "public_access": {},
                "connections": [],
                "endpoints": [],
                "workers": [],
                "worker_connections": [],
                "worker_counts": {"online": 0, "offline": 0, "total": 0},
                "waiting_requests": 0,
                "average_request_seconds": 0.2,
                "average_output_tokens_per_second": 20.0,
                "total_output_tokens_per_second": 20.0,
                "total_in_flight_requests": 0,
                "rendered_at": "12:06:00",
                "error_count": 0,
                "canonical_models": [],
            }
        if path == "/api/router/history":
            return {
                "points": [
                    {
                        "timestamp": "2026-05-01T12:00:00Z",
                        "epoch_seconds": 1.0,
                        "in_flight_requests": 1,
                        "throughput": 999.0,
                    },
                    {
                        "timestamp": "2026-05-01T12:06:00Z",
                        "epoch_seconds": 2.0,
                        "in_flight_requests": 2,
                        "throughput": 999.0,
                    },
                ],
                "window_seconds": 3600,
            }
        if path == "/api/router/requests":
            return {
                "max_entries": 1000,
                "total_retained": 2,
                "requests": [
                    {
                        "id": "req-1",
                        "timestamp": "2026-05-01T11:58:00Z",
                        "status": "completed",
                        "status_label": "Completed",
                        "completion_tokens": 120,
                        "completion_tokens_label": "120",
                        "duration_seconds": 3.0,
                        "duration_label": "3.00s",
                    },
                    {
                        "id": "req-2",
                        "timestamp": "2026-05-01T12:05:00Z",
                        "status": "completed",
                        "status_label": "Completed",
                        "completion_tokens": 30,
                        "completion_tokens_label": "30",
                        "duration_seconds": 1.0,
                        "duration_label": "1.00s",
                    },
                ],
            }
        raise AssertionError(f"Unexpected backend path: {path}")

    monkeypatch.setattr("ui_app._fetch_backend_json", _fake_fetch)
    monkeypatch.setattr("ui_app._fetch_backend_json_optional", lambda path, default: _fake_fetch(path))
    app = create_router_ui_app()
    with TestClient(app) as client:
        live = client.get("/router/live")
        assert live.status_code == 200
        payload = live.json()
        assert payload["metrics_history"]["points"][0]["throughput"] == 0.2
        assert payload["metrics_history"]["points"][1]["throughput"] == 0.25
