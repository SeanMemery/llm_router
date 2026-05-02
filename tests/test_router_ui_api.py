from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

import app as router_app
from app import create_router_app
from modal_worker_store import ModalWorkerLaunchRecord, ModalWorkerLaunchStatus, save_modal_worker_launches
from paths import ArtifactLayout
from worker_registry import LLMWorkerRecord, LLMWorkerStatus, LLMWorkerTransportMode


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
        "model: shared-test-model\nbase_url: http://router.example:8788\ntimeout: 30\n",
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


def test_router_panel_and_history_endpoints_return_cached_payloads(tmp_path: Path) -> None:
    app = create_router_app(layout=_layout(tmp_path))
    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 404

        errors_page = client.get("/router/errors")
        assert errors_page.status_code == 404

        panel = client.get("/api/router/panel")
        assert panel.status_code == 200
        panel_payload = panel.json()
        assert panel_payload["base_url"] == "http://router.example:8788"
        assert "total_in_flight_requests" in panel_payload
        assert "total_output_tokens_per_second" in panel_payload

        history = client.get("/api/router/history")
        assert history.status_code == 200
        history_payload = history.json()
        assert history_payload["window_seconds"] == 3600
        assert history_payload["throughput_unit"] == "tokens_per_second"
        assert isinstance(history_payload["points"], list)

def test_modal_launch_reconcile_marks_stopped_when_modal_app_is_not_live(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = _layout(tmp_path)
    log_path = layout.data_root / "dashboard" / "modal_workers" / "launch-1.log"
    state_path = layout.data_root / "dashboard" / "modal_workers" / "launch-1.state.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("modal log\n", encoding="utf-8")
    state_path.write_text('{"app_id":"ap-stopped"}\n', encoding="utf-8")
    started_at = datetime.now(UTC)
    launch = ModalWorkerLaunchRecord(
        launch_id="launch-1",
        display_name="modal-abc123",
        model="Qwen/Qwen2.5-7B-Instruct",
        duration_minutes=30,
        started_at=started_at,
        expires_at=started_at + timedelta(minutes=30),
        pid=12345,
        worker_id="worker-1",
        log_path=str(log_path),
        command=["python", "modal_worker.py"],
        status=ModalWorkerLaunchStatus.running,
        metadata={
            "launcher": "modal_worker",
            "modal_app_id": "ap-stopped",
        },
    )
    save_modal_worker_launches(layout, [launch])
    monkeypatch.setattr(router_app, "_modal_app_state_map", lambda _layout: {"ap-stopped": "stopped"})
    worker = LLMWorkerRecord(
        worker_id="worker-1",
        display_name="modal-abc123",
        hostname="host",
        endpoint_base_url=None,
        model="Qwen/Qwen2.5-7B-Instruct",
        local_model_path=None,
        local_model_root=None,
        api_key=None,
        max_concurrent_requests=8,
        router_max_concurrent_requests=None,
        router_priority=3,
        supports_image_inputs=False,
        transport_mode=LLMWorkerTransportMode.pull_request,
        metadata={
            "launcher": "modal_worker",
            "launch_id": "launch-1",
            "modal_app_id": "ap-stopped",
        },
        status=LLMWorkerStatus.online,
        registered_at=started_at,
        last_heartbeat_at=started_at,
        disconnected_at=None,
        last_error=None,
    )
    launches = router_app._reconcile_modal_worker_launches(layout=layout, workers=[worker])
    assert launches == []
    assert not log_path.exists()
    assert not state_path.exists()
