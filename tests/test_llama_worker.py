from __future__ import annotations

from pathlib import Path

import pytest
import httpx

from llama_worker import (
    _is_retryable_router_error,
    _is_unknown_worker_error,
    _resolve_model_identity,
)


def test_test_mode_does_not_require_model_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    model_path, model_name, local_model_root = _resolve_model_identity(
        test_mode=True,
        model_path=None,
        model_dir=str(model_dir),
        model_file=None,
        model_name="queue-probe-model",
    )

    assert model_path is None
    assert model_name == "queue-probe-model"
    assert local_model_root == str(model_dir)


def test_test_mode_uses_default_model_name_when_none_provided() -> None:
    model_path, model_name, local_model_root = _resolve_model_identity(
        test_mode=True,
        model_path=None,
        model_dir=None,
        model_file=None,
        model_name=None,
    )

    assert model_path is None
    assert model_name == "test-llama-worker"
    assert local_model_root is None


def test_normal_mode_requires_real_model_files(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Set LLAMA_MODEL_PATH or LLAMA_MODEL_DIR"):
        _resolve_model_identity(
            test_mode=False,
            model_path=None,
            model_dir=None,
            model_file=None,
            model_name=None,
        )


def test_unknown_worker_error_detection() -> None:
    request = httpx.Request("POST", "http://router.invalid/api/v1/workers/abc/requests/poll")
    response = httpx.Response(404, request=request)
    exc = httpx.HTTPStatusError("not found", request=request, response=response)
    assert _is_unknown_worker_error(exc) is True
    assert _is_retryable_router_error(exc) is False


def test_retryable_router_error_detection_for_502() -> None:
    request = httpx.Request("POST", "http://router.invalid/api/v1/workers/abc/requests/poll")
    response = httpx.Response(502, request=request)
    exc = httpx.HTTPStatusError("bad gateway", request=request, response=response)
    assert _is_unknown_worker_error(exc) is False
    assert _is_retryable_router_error(exc) is True
