from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app import create_router_app
from paths import ArtifactLayout


def _layout(tmp_path: Path) -> ArtifactLayout:
    project_root = tmp_path / "project"
    config_root = project_root / "configs"
    data_root = project_root / "data"
    evaluations_root = project_root / "evaluations"
    runs_root = project_root / "runs"
    reports_root = project_root / "reports"
    for path in (config_root, data_root, evaluations_root, runs_root, reports_root):
        path.mkdir(parents=True, exist_ok=True)
    (config_root / "LLM.yaml").write_text(
        "model: shared-test-model\nbase_url: http://example.invalid\ntimeout: 30\n",
        encoding="utf-8",
    )
    return ArtifactLayout(
        project_root=project_root,
        config_root=config_root,
        environment_manifest_root=config_root,
        data_root=data_root,
        evaluations_root=evaluations_root,
        runs_root=runs_root,
        reports_root=reports_root,
    )


def test_pull_request_worker_receives_and_completes_router_requests(tmp_path: Path) -> None:
    app = create_router_app(layout=_layout(tmp_path))
    with TestClient(app) as client:
        register = client.post(
            "/api/v1/workers/register",
            json={
                "display_name": "test-worker",
                "hostname": "test-host",
                "endpoint_base_url": None,
                "model": "test-model",
                "transport_mode": "pull-request",
                "max_concurrent_requests": 1,
                "supports_image_inputs": False,
                "metadata": {"launcher": "pytest"},
            },
        )
        assert register.status_code == 200
        worker_id = register.json()["worker"]["worker_id"]

        models = client.get("/v1/models")
        assert models.status_code == 200
        assert any(item["id"] == "test-model" for item in models.json()["data"])

        result: dict[str, object] = {}

        def _router_call() -> None:
            result["response"] = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello from router test"}],
                    "stream": False,
                },
            )

        router_thread = threading.Thread(target=_router_call, daemon=True)
        router_thread.start()

        request_payload: dict[str, object] | None = None
        for _ in range(50):
            poll = client.post(
                f"/api/v1/workers/{worker_id}/requests/poll",
                json={"max_requests": 1},
            )
            assert poll.status_code == 200
            requests = poll.json()["requests"]
            if requests:
                request_payload = requests[0]
                break
            time.sleep(0.05)

        assert request_payload is not None
        request_id = str(request_payload["request_id"])
        queued_payload = request_payload["payload"]
        assert queued_payload["model"] == "test-model"

        complete = client.post(
            f"/api/v1/workers/{worker_id}/requests/{request_id}/complete",
            json={
                "response_body": {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                }
            },
        )
        assert complete.status_code == 200

        router_thread.join(timeout=5.0)
        assert "response" in result
        response = result["response"]
        assert hasattr(response, "status_code")
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "ok"
        assert body["router"]["worker_request_id"] == request_id
        assert response.headers["X-LLM-Model"] == "test-model"


def test_pull_request_worker_failure_preserves_worker_status_code(tmp_path: Path) -> None:
    app = create_router_app(layout=_layout(tmp_path))
    with TestClient(app) as client:
        register = client.post(
            "/api/v1/workers/register",
            json={
                "display_name": "test-worker",
                "hostname": "test-host",
                "endpoint_base_url": None,
                "model": "test-model",
                "transport_mode": "pull-request",
                "max_concurrent_requests": 1,
                "supports_image_inputs": False,
                "metadata": {"launcher": "pytest"},
            },
        )
        assert register.status_code == 200
        worker_id = register.json()["worker"]["worker_id"]

        result: dict[str, object] = {}

        def _router_call() -> None:
            result["response"] = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "bad request"}],
                    "stream": False,
                },
            )

        router_thread = threading.Thread(target=_router_call, daemon=True)
        router_thread.start()

        request_payload: dict[str, object] | None = None
        for _ in range(50):
            poll = client.post(
                f"/api/v1/workers/{worker_id}/requests/poll",
                json={"max_requests": 1},
            )
            assert poll.status_code == 200
            requests = poll.json()["requests"]
            if requests:
                request_payload = requests[0]
                break
            time.sleep(0.05)

        assert request_payload is not None
        request_id = str(request_payload["request_id"])
        fail = client.post(
            f"/api/v1/workers/{worker_id}/requests/{request_id}/fail",
            json={"error": "Worker local backend rejected request payload", "status_code": 400},
        )
        assert fail.status_code == 200

        router_thread.join(timeout=5.0)
        assert "response" in result
        response = result["response"]
        assert hasattr(response, "status_code")
        assert response.status_code == 400
        assert response.json()["detail"] == "Worker local backend rejected request payload"
