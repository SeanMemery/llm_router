from __future__ import annotations

from pathlib import Path

import pytest

from vllm_worker import (
    DEFAULT_VLLM_MAX_CONCURRENCY,
    _build_vllm_command,
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

    command = _build_vllm_command("Qwen/Qwen2.5-7B-Instruct", host="0.0.0.0", port=8080)

    assert "--served-model-name" in command
    assert "demo-alias" in command
    assert "--max-model-len" in command
    assert "8192" in command
    assert "--max-num-seqs" in command
    assert str(DEFAULT_VLLM_MAX_CONCURRENCY) in command
