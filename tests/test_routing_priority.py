from __future__ import annotations

import asyncio

import httpx

from dashboard.llm_router import (
    DashboardLLMRouter,
    LLMRouterConnectionConfig,
    LLMRouterConnectionSnapshot,
    LLMRouterSnapshot,
)


def test_lower_numeric_priority_is_selected_before_worker_preference() -> None:
    observed: dict[str, str] = {}

    def _client_factory(config: LLMRouterConnectionConfig, timeout_seconds: float | None) -> httpx.AsyncClient:
        async def _handler(request: httpx.Request) -> httpx.Response:
            observed["selected_connection_id"] = config.connection_id
            payload = {
                "id": "chatcmpl-manual",
                "object": "chat.completion",
                "created": 1,
                "model": config.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "manual"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            return httpx.Response(200, request=request, json=payload)

        return httpx.AsyncClient(
            transport=httpx.MockTransport(_handler),
            base_url="http://manual.invalid",
            timeout=timeout_seconds,
            headers={"Content-Type": "application/json"},
        )

    async def _run() -> None:
        router = DashboardLLMRouter(
            connections=[
                LLMRouterConnectionConfig(
                    connection_id="manual:preferred",
                    base_url="http://manual.invalid",
                    model="Qwen3.6-35B-A3B-UD-Q4_K_M",
                    source_kind="manual",
                    starred=False,
                    priority=1,
                    max_concurrent_requests=1,
                    supports_image_inputs=True,
                ),
                LLMRouterConnectionConfig(
                    connection_id="worker:secondary",
                    base_url="worker://secondary",
                    model="Qwen3.6-35B-A3B-UD-Q4_K_M",
                    source_kind="worker",
                    worker_id="secondary",
                    transport_mode="pull-request",
                    starred=False,
                    priority=2,
                    max_concurrent_requests=1,
                    supports_image_inputs=True,
                ),
            ],
            initial_snapshot=LLMRouterSnapshot(
                connections=[
                    LLMRouterConnectionSnapshot(
                        connection_id="manual:preferred",
                        base_url="http://manual.invalid",
                        model="Qwen3.6-35B-A3B-UD-Q4_K_M",
                        source_kind="manual",
                        starred=False,
                        priority=1,
                        max_concurrent_requests=1,
                        supports_image_inputs=True,
                        active=True,
                    ),
                    LLMRouterConnectionSnapshot(
                        connection_id="worker:secondary",
                        base_url="worker://secondary",
                        model="Qwen3.6-35B-A3B-UD-Q4_K_M",
                        source_kind="worker",
                        worker_id="secondary",
                        transport_mode="pull-request",
                        starred=False,
                        priority=2,
                        max_concurrent_requests=1,
                        supports_image_inputs=True,
                        active=True,
                    ),
                ]
            ),
            client_factory=_client_factory,
            worker_request_handler=lambda config, payload: asyncio.sleep(
                0,
                result={
                    "id": "chatcmpl-worker",
                    "object": "chat.completion",
                    "created": 1,
                    "model": config.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "worker"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            ),
        )
        try:
            result = await router.chat_completion(
                {
                    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                }
            )
        finally:
            await router.aclose()

        assert result.body["choices"][0]["message"]["content"] == "manual"
        assert result.connection_id == "manual:preferred"
        assert observed["selected_connection_id"] == "manual:preferred"

    asyncio.run(_run())
