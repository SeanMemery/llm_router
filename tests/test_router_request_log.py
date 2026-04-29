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


def _register_pull_worker(client: TestClient) -> str:
    response = client.post(
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
    assert response.status_code == 200
    return str(response.json()["worker"]["worker_id"])


def _wait_for_worker_request(client: TestClient, worker_id: str) -> dict[str, object]:
    for _ in range(50):
        poll = client.post(
            f"/api/v1/workers/{worker_id}/requests/poll",
            json={"max_requests": 1},
        )
        assert poll.status_code == 200
        requests = poll.json()["requests"]
        if requests:
            return requests[0]
        time.sleep(0.05)
    raise AssertionError("worker request was not queued in time")


def test_router_request_log_persists_completed_requests(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    app = create_router_app(layout=layout)
    with TestClient(app) as client:
        worker_id = _register_pull_worker(client)
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

        queued = _wait_for_worker_request(client, worker_id)
        request_id = str(queued["request_id"])
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
        response = result["response"]
        assert hasattr(response, "status_code")
        assert response.status_code == 200

        request_index = client.get("/api/router/requests")
        assert request_index.status_code == 200
        requests_payload = request_index.json()
        assert requests_payload["total_retained"] == 1
        summary = requests_payload["requests"][0]
        assert summary["status"] == "completed"
        assert summary["prompt_tokens"] == 3
        assert summary["completion_tokens"] == 1
        assert summary["http_status_code"] == 200

        detail = client.get(f"/api/router/requests/{summary['id']}")
        assert detail.status_code == 200
        detail_payload = detail.json()
        assert detail_payload["usage"]["prompt_tokens"] == 3
        assert detail_payload["usage"]["completion_tokens"] == 1
        assert detail_payload["connection"]["worker_request_id"] == request_id
        assert detail_payload["response_headers"]["X-LLM-Model"] == "test-model"

    index_path = layout.data_root / "dashboard" / "llm_router_requests.json"
    detail_path = layout.data_root / "dashboard" / "llm_router_requests" / f"{summary['id']}.json"
    assert index_path.is_file()
    assert detail_path.is_file()


def test_router_request_log_persists_failed_requests(tmp_path: Path) -> None:
    app = create_router_app(layout=_layout(tmp_path))
    with TestClient(app) as client:
        worker_id = _register_pull_worker(client)
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

        queued = _wait_for_worker_request(client, worker_id)
        request_id = str(queued["request_id"])
        failed = client.post(
            f"/api/v1/workers/{worker_id}/requests/{request_id}/fail",
            json={"error": "Worker local backend rejected request payload", "status_code": 400},
        )
        assert failed.status_code == 200

        router_thread.join(timeout=5.0)
        response = result["response"]
        assert hasattr(response, "status_code")
        assert response.status_code == 400

        request_index = client.get("/api/router/requests")
        assert request_index.status_code == 200
        summary = request_index.json()["requests"][0]
        assert summary["status"] == "failed"
        assert summary["http_status_code"] == 400
        assert summary["completion_tokens"] is None

        detail = client.get(f"/api/router/requests/{summary['id']}")
        assert detail.status_code == 200
        detail_payload = detail.json()
        assert detail_payload["error"] == "Worker local backend rejected request payload"
        assert detail_payload["status"] == "failed"
