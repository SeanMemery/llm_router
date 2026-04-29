from __future__ import annotations

import asyncio

import pytest

from dashboard.llm_router import DashboardLLMRouter, LLMRouterConnectionConfig


def test_worker_usage_percentage_tracks_time_with_active_requests() -> None:
    now = 100.0

    def _time_fn() -> float:
        return now

    async def _exercise() -> None:
        nonlocal now
        router = DashboardLLMRouter(
            connections=[
                LLMRouterConnectionConfig(
                    connection_id="worker:alpha",
                    base_url="worker://alpha",
                    model="test-model",
                    source_kind="worker",
                    worker_id="alpha",
                    transport_mode="pull-request",
                    max_concurrent_requests=1,
                    supports_image_inputs=False,
                )
            ],
            worker_request_handler=lambda _config, _payload: asyncio.sleep(0),  # unused
            time_fn=_time_fn,
        )
        runtime = router._runtime_for_id("worker:alpha")
        now = 110.0
        snapshot = router.snapshot()
        assert snapshot.connections[0].usage_percentage == 0.0

        acquired = await router._acquire_runtime()
        assert acquired is runtime
        now = 115.0
        snapshot = router.snapshot()
        assert snapshot.connections[0].usage_percentage == pytest.approx(33.33333333333333)

        await router._release_runtime(runtime)
        now = 120.0
        snapshot = router.snapshot()
        assert snapshot.connections[0].usage_percentage == pytest.approx(25.0)

        await router.aclose()

    asyncio.run(_exercise())
