from __future__ import annotations

from dashboard.llm_router import (
    GLOBAL_MIN_COMPLETION_TOKENS,
    DashboardLLMRouter,
    LLMRouterConnectionConfig,
)


def test_router_normalization_raises_low_max_tokens() -> None:
    config = LLMRouterConnectionConfig(
        connection_id="conn",
        base_url="https://router.example/v1",
        model="demo-model",
        max_concurrent_requests=1,
    )

    payload = DashboardLLMRouter._normalize_chat_payload_for_connection(
        {"model": "demo-model", "messages": [], "max_tokens": 2048},
        config,
    )

    assert payload["max_tokens"] == GLOBAL_MIN_COMPLETION_TOKENS


def test_router_normalization_injects_default_max_tokens() -> None:
    config = LLMRouterConnectionConfig(
        connection_id="conn",
        base_url="https://api.openai.com/v1",
        model="gpt-5-mini",
        max_concurrent_requests=1,
    )

    payload = DashboardLLMRouter._normalize_chat_payload_for_connection(
        {"model": "gpt-5-mini", "messages": []},
        config,
    )

    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == GLOBAL_MIN_COMPLETION_TOKENS


def test_router_normalization_forces_qwen_sampling_settings() -> None:
    config = LLMRouterConnectionConfig(
        connection_id="conn",
        base_url="https://router.example/v1",
        model="Qwen3.6-35B-A3B-UD-Q4_K_M",
        max_concurrent_requests=1,
    )

    payload = DashboardLLMRouter._normalize_chat_payload_for_connection(
        {
            "model": "Qwen3.6-35B-A3B-UD-Q4_K_M",
            "messages": [],
            "temperature": 0.2,
            "top_p": 0.3,
            "presence_penalty": 0.1,
            "extra_body": {"top_k": 5, "min_p": 0.1},
        },
        config,
    )

    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["presence_penalty"] == 1.5
    assert payload["extra_body"]["top_k"] == 20
    assert payload["extra_body"]["min_p"] == 0.1
