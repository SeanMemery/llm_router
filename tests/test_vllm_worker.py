from __future__ import annotations

from pathlib import Path

import pytest

from vllm_worker import (
    DEFAULT_VLLM_MAX_CONCURRENCY,
    DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS,
    _build_vllm_command,
    _default_vllm_moe_backend,
    _default_vllm_reasoning_parser,
    _normalize_router_url,
    _resolve_model_identity,
)


def test_test_mode_does_not_require_model_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    model_target, model_name, local_model_root = _resolve_model_identity(
        test_mode=True,
        model_path=None,
        model_dir=str(model_dir),
        model_file=None,
        model_name="queue-probe-model",
    )

    assert model_target is None
    assert model_name == "queue-probe-model"
    assert local_model_root == str(model_dir)


def test_normal_mode_accepts_existing_model_file(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_text("demo", encoding="utf-8")

    model_target, model_name, local_model_root = _resolve_model_identity(
        test_mode=False,
        model_path=str(model_path),
        model_dir=None,
        model_file=None,
        model_name=None,
    )

    assert model_target == str(model_path)
    assert model_name == "model"
    assert local_model_root is None


def test_normal_mode_accepts_remote_model_reference() -> None:
    model_target, model_name, local_model_root = _resolve_model_identity(
        test_mode=False,
        model_path="Qwen/Qwen2.5-7B-Instruct",
        model_dir=None,
        model_file=None,
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )

    assert model_target == "Qwen/Qwen2.5-7B-Instruct"
    assert model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert local_model_root is None


def test_build_vllm_command_maps_parallel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_MODEL_NAME", "demo-alias")
    monkeypatch.setenv("LLAMA_CTX_SIZE", "8192")
    monkeypatch.setenv("LLAMA_MAX_CONCURRENCY", str(DEFAULT_VLLM_MAX_CONCURRENCY))
    monkeypatch.setenv("VLLM_KV_CACHE_DTYPE", "fp8")

    command = _build_vllm_command("Qwen/Qwen2.5-7B-Instruct", host="0.0.0.0", port=8080)

    assert "--served-model-name" in command
    assert "demo-alias" in command
    assert "--max-model-len" in command
    assert "8192" in command
    assert "--max-num-seqs" in command
    assert str(DEFAULT_VLLM_MAX_CONCURRENCY) in command
    assert "--kv-cache-dtype" in command
    assert "fp8" in command


def test_build_vllm_command_applies_qwen36_nvfp4_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_MODEL_NAME", "RedHatAI/Qwen3.6-35B-A3B-NVFP4")

    command = _build_vllm_command("RedHatAI/Qwen3.6-35B-A3B-NVFP4", host="0.0.0.0", port=8080)

    assert "--reasoning-parser" in command
    assert "qwen3" in command
    assert "--moe-backend" in command
    assert "marlin" in command


def test_build_vllm_command_allows_override_of_model_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_MODEL_NAME", "RedHatAI/Qwen3.6-35B-A3B-NVFP4")
    monkeypatch.setenv("VLLM_REASONING_PARSER", "custom-parser")
    monkeypatch.setenv("VLLM_MOE_BACKEND", "cutlass")

    command = _build_vllm_command("RedHatAI/Qwen3.6-35B-A3B-NVFP4", host="0.0.0.0", port=8080)

    assert "--reasoning-parser" in command
    assert "custom-parser" in command
    assert "--moe-backend" in command
    assert "cutlass" in command


def test_vllm_model_default_helpers() -> None:
    assert _default_vllm_reasoning_parser("Qwen/Qwen3.6-35B-A3B") == "qwen3"
    assert _default_vllm_reasoning_parser("RedHatAI/Qwen3.6-35B-A3B-NVFP4") == "qwen3"
    assert _default_vllm_reasoning_parser("meta-llama/Llama-3.3-70B-Instruct") is None
    assert _default_vllm_moe_backend("RedHatAI/Qwen3.6-35B-A3B-NVFP4") == "marlin"
    assert _default_vllm_moe_backend("Qwen/Qwen3.6-35B-A3B-FP8") is None


def test_default_startup_timeout_is_raised_for_slow_first_boots() -> None:
    assert DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS == 900.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://router:8788", "http://router:8788"),
        ("http://router:8788/v1", "http://router:8788"),
        ("https://router.example/public", "https://router.example"),
        ("https://router.example/public/v1", "https://router.example"),
    ],
)
def test_normalize_router_url(raw: str, expected: str) -> None:
    assert _normalize_router_url(raw) == expected
