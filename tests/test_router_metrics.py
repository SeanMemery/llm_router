from __future__ import annotations

from app import _router_overview_metrics
from dashboard.llm_router import (
    LLMRouterConnectionSnapshot,
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
