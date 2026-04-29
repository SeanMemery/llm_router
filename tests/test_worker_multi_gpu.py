from __future__ import annotations

from pathlib import Path

import pytest

from worker_multi_gpu import (
    _parse_gpu_indices,
    _public_base_url_for_gpu,
    _resolve_model_path,
)


def test_parse_gpu_indices_discards_empty_values() -> None:
    assert _parse_gpu_indices("0, 1, ,2") == ["0", "1", "2"]


def test_public_base_url_template_expands_gpu_and_port() -> None:
    assert (
        _public_base_url_for_gpu(
            template="http://worker-{gpu}.svc:{port}",
            gpu_index="3",
            port=8091,
            worker_index=1,
        )
        == "http://worker-3.svc:8091"
    )


def test_public_base_url_requires_template_for_multiple_direct_workers() -> None:
    with pytest.raises(SystemExit, match="must contain"):
        _public_base_url_for_gpu(
            template="http://worker.service:8080",
            gpu_index="1",
            port=8081,
            worker_index=1,
        )


def test_resolve_model_path_prefers_download_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "models" / "demo.gguf"
    monkeypatch.setattr("worker_multi_gpu._download_target_path", lambda: target)
    assert _resolve_model_path() == target


def test_resolve_model_path_uses_explicit_model_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_text("demo", encoding="utf-8")
    monkeypatch.setenv("LLAMA_MODEL_PATH", str(model_path))
    monkeypatch.delenv("LLAMA_MODEL_DIR", raising=False)
    monkeypatch.setattr("worker_multi_gpu._download_target_path", lambda: None)
    assert _resolve_model_path() == model_path
