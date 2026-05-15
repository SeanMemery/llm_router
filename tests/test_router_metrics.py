from __future__ import annotations

import time

from datetime import datetime, timezone

from app import (
    _request_log_throughput_by_connection,
    _router_overview_metrics,
    _serialize_router_connection_snapshot,
)
from dashboard.llm_router import (
    LLMRouterConnectionSnapshot,
    PER_MODEL_THROUGHPUT_WINDOW_SECONDS,
    LLMRouterSnapshot,
    LLMRouterTelemetry,
)


def _telemetry(*, completion_tokens: int, request_seconds: float, concurrent_requests_seen: int) -> LLMRouterTelemetry:
    telemetry = LLMRouterTelemetry()
    telemetry.record_request(
        prompt_tokens=100,
        completion_tokens=completion_tokens,
        request_duration_seconds=request_seconds,
        concurrent_requests_seen=concurrent_requests_seen,
    )
    return telemetry


def test_per_model_tokens_per_second_uses_total_output_over_last_10_minutes(monkeypatch) -> None:
    telemetry = LLMRouterTelemetry()
    now = 1_800.0
    monkeypatch.setattr(time, "time", lambda: now)
    telemetry.record_request(
        prompt_tokens=10,
        completion_tokens=240,
        request_duration_seconds=4.0,
        concurrent_requests_seen=1,
        timestamp_epoch_seconds=now - 120.0,
    )
    telemetry.record_request(
        prompt_tokens=10,
        completion_tokens=60,
        request_duration_seconds=2.0,
        concurrent_requests_seen=1,
        timestamp_epoch_seconds=now - 30.0,
    )
    telemetry.record_request(
        prompt_tokens=10,
        completion_tokens=9_999,
        request_duration_seconds=1.0,
        concurrent_requests_seen=1,
        timestamp_epoch_seconds=now - (PER_MODEL_THROUGHPUT_WINDOW_SECONDS + 1.0),
    )

    assert telemetry.average_output_tokens_per_second == 0.5


def test_serialized_connection_uses_request_log_throughput_override() -> None:
    connection = LLMRouterConnectionSnapshot(
        connection_id="conn-1",
        base_url="http://worker.example",
        model="demo-model",
        max_concurrent_requests=4,
        telemetry=_telemetry(
            completion_tokens=400,
            request_seconds=2.0,
            concurrent_requests_seen=1,
        ),
    )
    throughput_by_connection = _request_log_throughput_by_connection(
        [
            {
                "timestamp": "2026-05-03T13:58:00Z",
                "completion_tokens": 120,
                "duration_seconds": 2.0,
                "connection": {"connection_id": "conn-1"},
            },
            {
                "timestamp": "2026-05-03T13:54:00Z",
                "completion_tokens": 180,
                "duration_seconds": 3.0,
                "connection": {"connection_id": "conn-1"},
            },
            {
                "timestamp": "2026-05-03T13:40:00Z",
                "completion_tokens": 999,
                "duration_seconds": 1.0,
                "connection": {"connection_id": "conn-1"},
            },
        ],
        window_end=datetime(2026, 5, 3, 14, 0, 0, tzinfo=timezone.utc),
    )

    serialized = _serialize_router_connection_snapshot(
        connection,
        throughput_by_connection_id=throughput_by_connection,
    )

    assert serialized["telemetry"]["average_output_tokens_per_second"] == 0.5


def test_request_log_throughput_by_connection_supports_summary_entries() -> None:
    throughput_by_connection = _request_log_throughput_by_connection(
        [
            {
                "timestamp": "2026-05-03T13:58:00Z",
                "completion_tokens": 120,
                "duration_seconds": 2.0,
                "base_url": "worker://worker-1",
                "selected_model": "demo-model",
                "source_kind": "worker",
                "worker_id": "worker-1",
            },
            {
                "timestamp": "2026-05-03T13:54:00Z",
                "completion_tokens": 180,
                "duration_seconds": 3.0,
                "base_url": "http://example.local:8000",
                "selected_model": "demo-model",
                "source_kind": "manual",
            },
        ],
        window_end=datetime(2026, 5, 3, 14, 0, 0, tzinfo=timezone.utc),
    )

    assert throughput_by_connection["worker:worker-1"] == 0.2
    assert throughput_by_connection["http-example-local-8000-demo-model"] == 0.3


def test_router_overview_metrics_only_count_active_connections_for_dynamic_metrics() -> None:
    active = LLMRouterConnectionSnapshot(
        connection_id="active",
        base_url="http://active.example",
        model="active-model",
        max_concurrent_requests=4,
        active=True,
        manually_disabled=False,
        in_flight_requests=2,
        telemetry=_telemetry(
            completion_tokens=200,
            request_seconds=10.0,
            concurrent_requests_seen=2,
        ),
    )
    inactive = LLMRouterConnectionSnapshot(
        connection_id="inactive",
        base_url="http://inactive.example",
        model="inactive-model",
        max_concurrent_requests=8,
        active=False,
        manually_disabled=False,
        in_flight_requests=7,
        telemetry=_telemetry(
            completion_tokens=10_000,
            request_seconds=1.0,
            concurrent_requests_seen=8,
        ),
    )
    snapshot = LLMRouterSnapshot(connections=[active, inactive])

    metrics = _router_overview_metrics(snapshot)

    assert metrics["total_output_tokens_per_second"] == 40.0
    assert metrics["average_output_tokens_per_second"] == 20.0
    assert metrics["average_request_seconds"] == 10.0
    assert metrics["total_in_flight_requests"] == 2
    assert metrics["total_prompt_tokens"] == 200
    assert metrics["total_prompt_tokens_label"] == "200"
    assert metrics["total_completion_tokens"] == 10_200
    assert metrics["total_completion_tokens_label"] == "10.20K"
