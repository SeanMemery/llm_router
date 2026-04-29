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
        assert "modal_launches" in panel_payload
        assert "modal_model_options" in panel_payload

        history = client.get("/api/router/history")
        assert history.status_code == 200
        history_payload = history.json()
        assert history_payload["window_seconds"] == 3600
        assert history_payload["throughput_unit"] == "tokens_per_second"
        assert isinstance(history_payload["points"], list)


def test_modal_worker_launch_and_stop_round_trip(tmp_path: Path, monkeypatch) -> None:
    app = create_router_app(layout=_layout(tmp_path))

    class _FakeProcess:
        def __init__(self, *_args, **_kwargs) -> None:
            self.pid = 43210

    killed: list[int] = []
    stopped_apps: list[str] = []

    monkeypatch.setattr(router_app.importlib.util, "find_spec", lambda name: object() if name == "modal" else None)
    monkeypatch.setattr(router_app.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())
    monkeypatch.setattr(router_app, "_pid_is_alive", lambda pid: True)
    monkeypatch.setattr(router_app, "_router_public_access_payload", lambda layout: {})
    monkeypatch.setattr(router_app.os, "killpg", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(
        router_app,
        "_stop_modal_launch_app",
        lambda *, layout, launch_id, modal_app_id: (stopped_apps.append(modal_app_id or "") or True, None),
    )

    with TestClient(app) as client:
        start = client.post(
            "/router/modal-workers/start",
            data={
                "next": "/",
                "modal_model": "Qwen/Qwen3.6-35B-A3B-FP8",
                "modal_duration_minutes": "45",
            },
            follow_redirects=False,
        )
        assert start.status_code == 303

        panel = client.get("/api/router/panel")
        assert panel.status_code == 200
        payload = panel.json()
        assert len(payload["modal_launches"]) == 1
        launch = payload["modal_launches"][0]
        assert launch["display_name"].startswith("modal-")
        assert launch["model"] == "Qwen/Qwen3.6-35B-A3B-FP8"
        assert launch["status"] == "starting"
        assert launch["metadata"]["execution_mode"] == "vllm"

        launch_state_path = tmp_path / "project" / "data" / "dashboard" / "modal_workers" / f"{launch['launch_id']}.state.json"
        launch_state_path.parent.mkdir(parents=True, exist_ok=True)
        launch_state_path.write_text(
            '{"app_id":"ap-test-123","execution_mode":"vllm","state":"registered","worker_id":"worker-123"}\n',
            encoding="utf-8",
        )

        stop = client.post(
            f"/router/modal-workers/{launch['launch_id']}/stop",
            data={"next": "/"},
            follow_redirects=False,
        )
        assert stop.status_code == 303
        assert killed == [43210]
        assert stopped_apps == ["ap-test-123"]

        panel_after = client.get("/api/router/panel")
        assert panel_after.status_code == 200
        assert panel_after.json()["modal_launches"] == []


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
