from __future__ import annotations

from fastapi.testclient import TestClient

from ui_app import create_router_ui_app


def test_router_ui_home_includes_request_panel(monkeypatch) -> None:
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
                "worker_connections": [],
                "worker_counts": {"online": 0, "offline": 0, "total": 0},
                "modal_launches": [
                    {
                        "launch_id": "launch-1",
                        "display_name": "modal qwen",
                        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
                        "duration_minutes": 60,
                        "status": "running",
                        "worker_id": "worker-1",
                    },
                    {
                        "launch_id": "launch-2",
                        "display_name": "modal old",
                        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
                        "duration_minutes": 60,
                        "status": "stopped",
                        "worker_id": None,
                    },
                    {
                        "launch_id": "launch-3",
                        "display_name": "modal failed",
                        "model": "Qwen/Qwen2.5-32B-Instruct",
                        "duration_minutes": 60,
                        "status": "failed",
                        "worker_id": None,
                    }
                ],
                "modal_model_options": ["Qwen/Qwen3.6-35B-A3B-FP8"],
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
                        "completion_tokens": 20,
                        "prompt_tokens_label": "10",
                        "completion_tokens_label": "20",
                        "duration_seconds": 0.5,
                        "duration_label": "500ms",
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
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                "timing": {"router_request_seconds": 0.5, "upstream_request_seconds": 0.4},
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
                    "duration_label": "500ms",
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
        assert "Recent router traffic" not in home.text
        assert "/router/requests/req-1" in home.text
        assert "selected-model" not in home.text
        assert "s1804274-infk8s-40gb-router-worker-bjbwj-wnpwf" in home.text
        assert "Start Modal" in home.text
        assert "modal qwen" in home.text
        assert "modal old" not in home.text
        assert "modal failed" not in home.text
        assert "Automatic</p>" in home.text
        assert "No automatic router workers are connected." in home.text
        assert ">200<" not in home.text
        assert "2026-04-27 13:00:00" in home.text

        detail = client.get("/router/requests/req-1")
        assert detail.status_code == 200
        assert "Request req-1" in detail.text
        assert "Explain the route choice." in detail.text
        assert "The router picked the lowest-cost healthy backend." in detail.text
        assert "It selected the healthy worker with available capacity." in detail.text
        assert "selected-model" in detail.text
