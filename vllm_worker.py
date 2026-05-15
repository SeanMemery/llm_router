from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx


DEFAULT_VLLM_MAX_CONCURRENCY = 12
DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS = 900.0


def _sleep_until(deadline_seconds: float, *, terminate_flag: Callable[[], bool]) -> None:
    while not terminate_flag():
        remaining = deadline_seconds - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.25, remaining))


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


def _normalize_router_url(value: str) -> str:
    base = str(value or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/public/v1"):
        return base[: -len("/public/v1")]
    if base.endswith("/public"):
        return base[: -len("/public")]
    if base.endswith("/v1"):
        return base[: -len("/v1")]
    return base


def _normalize_model_reference(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.exists():
        return str(candidate)
    if "://" in text:
        return text
    if text.startswith("/") or text.startswith("."):
        raise SystemExit(f"Configured model path does not exist: {text}")
    return text


def _default_vllm_reasoning_parser(model_reference: str | None) -> str | None:
    candidate = str(model_reference or "").strip()
    if candidate.startswith("Qwen/Qwen3") or candidate.startswith("RedHatAI/Qwen3"):
        return "qwen3"
    return None


def _default_vllm_moe_backend(model_reference: str | None) -> str | None:
    candidate = str(model_reference or "").strip()
    if candidate == "RedHatAI/Qwen3.6-35B-A3B-NVFP4":
        return "marlin"
    return None


def _find_model_target(*, model_path: str | None, model_dir: str | None, model_file: str | None) -> str:
    normalized_model_path = _normalize_model_reference(model_path)
    if normalized_model_path:
        return normalized_model_path
    if not model_dir:
        raise SystemExit("Set LLAMA_MODEL_PATH, LLAMA_MODEL_DIR, or LLAMA_MODEL_NAME.")
    root = Path(model_dir).expanduser()
    if not root.is_dir():
        normalized = _normalize_model_reference(model_dir)
        if normalized:
            return normalized
        raise SystemExit(f"Configured LLAMA_MODEL_DIR does not exist: {root}")
    if model_file:
        candidate = root / model_file
        if not candidate.exists():
            raise SystemExit(f"Configured LLAMA_MODEL_FILE does not exist under LLAMA_MODEL_DIR: {candidate}")
        return str(candidate)
    gguf_candidates = [
        item
        for item in sorted(root.rglob("*.gguf"))
        if "mmproj" not in item.name.lower() and "projector" not in item.name.lower()
    ]
    if len(gguf_candidates) == 1:
        return str(gguf_candidates[0])
    if len(gguf_candidates) > 1:
        raise SystemExit(
            "Multiple GGUF files found. Set LLAMA_MODEL_FILE or LLAMA_MODEL_PATH explicitly: "
            + ", ".join(str(item.relative_to(root)) for item in gguf_candidates[:8])
        )
    return str(root)


def _resolve_model_identity(
    *,
    test_mode: bool,
    model_path: str | None,
    model_dir: str | None,
    model_file: str | None,
    model_name: str | None,
) -> tuple[str | None, str, str | None]:
    if test_mode:
        local_model_root = str(Path(model_dir).expanduser()) if model_dir else None
        resolved_name = (model_name or "test-vllm-worker").strip()
        if not resolved_name:
            raise SystemExit("Set --model-name or LLM_MODEL_NAME when using test mode.")
        return None, resolved_name, local_model_root

    normalized_model_name = _normalize_model_reference(model_name)
    resolved_target = _find_model_target(
        model_path=model_path,
        model_dir=model_dir,
        model_file=model_file,
    ) if (model_path or model_dir or model_file) else normalized_model_name
    if not resolved_target:
        raise SystemExit("Set LLAMA_MODEL_PATH, LLAMA_MODEL_DIR, or LLAMA_MODEL_NAME.")
    target_path = Path(resolved_target).expanduser()
    local_model_root = str(Path(model_dir).expanduser()) if model_dir else None
    resolved_name = (model_name or target_path.stem or target_path.name or resolved_target).strip()
    if not resolved_name:
        resolved_name = resolved_target
    return resolved_target, resolved_name, local_model_root


def _build_vllm_command(model_target: str, *, host: str, port: int) -> list[str]:
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model_target,
    ]
    model_name = _env("LLAMA_MODEL_NAME")
    if model_name:
        command.extend(["--served-model-name", model_name])
    ctx_size = _env("LLAMA_CTX_SIZE")
    if ctx_size:
        command.extend(["--max-model-len", ctx_size])
    parallel = _env("LLAMA_MAX_CONCURRENCY")
    if parallel:
        command.extend(["--max-num-seqs", parallel])
    download_dir = _env("LLAMA_MODEL_DIR")
    if download_dir and not Path(model_target).expanduser().exists():
        command.extend(["--download-dir", download_dir])
    gpu_memory = _env("VLLM_GPU_MEMORY_UTILIZATION")
    if gpu_memory:
        command.extend(["--gpu-memory-utilization", gpu_memory])
    kv_cache_dtype = _env("VLLM_KV_CACHE_DTYPE")
    if kv_cache_dtype:
        command.extend(["--kv-cache-dtype", kv_cache_dtype])
    tensor_parallel = _env("VLLM_TENSOR_PARALLEL_SIZE")
    if tensor_parallel:
        command.extend(["--tensor-parallel-size", tensor_parallel])
    reasoning_parser = _env("VLLM_REASONING_PARSER") or _default_vllm_reasoning_parser(
        _env("LLAMA_MODEL_NAME") or model_target
    )
    if reasoning_parser:
        command.extend(["--reasoning-parser", reasoning_parser])
    moe_backend = _env("VLLM_MOE_BACKEND") or _default_vllm_moe_backend(
        _env("LLAMA_MODEL_NAME") or model_target
    )
    if moe_backend:
        command.extend(["--moe-backend", moe_backend])
    if _env("VLLM_TRUST_REMOTE_CODE", "0") == "1":
        command.append("--trust-remote-code")
    extra_args = _env("VLLM_EXTRA_ARGS") or _env("LLAMA_EXTRA_ARGS")
    if extra_args:
        command.extend(shlex.split(extra_args))
    return command


def _wait_for_local_server(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(f"{base_url}/v1/models")
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                    return
            except Exception:
                time.sleep(1.0)
                continue
    raise SystemExit(f"Timed out waiting for local vLLM server readiness at {base_url}")


def _router_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = _env("LLM_ROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _is_unknown_worker_error(exc: Exception) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code == 404
        and "/api/v1/workers/" in str(exc.request.url)
    )


def _is_retryable_router_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


def _router_retry_delay_seconds(*, attempt: int, base_seconds: float = 1.0, cap_seconds: float = 15.0) -> float:
    bounded_attempt = max(0, min(attempt, 6))
    return min(cap_seconds, base_seconds * (2**bounded_attempt))


def _mock_chat_completion(payload: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    messages = payload.get("messages")
    prompt_preview = ""
    if isinstance(messages, list) and messages:
        last_message = messages[-1]
        if isinstance(last_message, dict):
            content = last_message.get("content")
            if isinstance(content, str):
                prompt_preview = content[:120]
            elif isinstance(content, list):
                prompt_preview = f"{len(content)} content parts"
    return {
        "id": f"chatcmpl-test-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"test worker response: {prompt_preview}".strip(),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": max(1, len(prompt_preview.split()) or 1),
            "completion_tokens": 4,
            "total_tokens": max(1, len(prompt_preview.split()) or 1) + 4,
        },
        "router": {
            "worker_execution_mode": "test",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-registering vLLM worker")
    parser.add_argument("--router-url", default=_env("LLM_ROUTER_URL"))
    parser.add_argument("--model-dir", default=_env("LLAMA_MODEL_DIR"))
    parser.add_argument("--model-path", default=_env("LLAMA_MODEL_PATH"))
    parser.add_argument("--model-file", default=_env("LLAMA_MODEL_FILE"))
    parser.add_argument("--model-name", default=_env("LLAMA_MODEL_NAME"))
    parser.add_argument("--public-base-url", default=_env("LLM_WORKER_PUBLIC_BASE_URL"))
    parser.add_argument("--display-name", default=_env("LLAMA_DISPLAY_NAME") or socket.gethostname())
    parser.add_argument("--host", default=_env("LLAMA_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(_env("LLAMA_PORT", "8080")))
    parser.add_argument("--heartbeat-seconds", type=float, default=float(_env("LLM_WORKER_HEARTBEAT_SECONDS", "15")))
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=float(_env("LLM_WORKER_STARTUP_TIMEOUT_SECONDS", str(DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS))),
    )
    parser.add_argument("--supports-image-inputs", action="store_true", default=_env("LLAMA_SUPPORTS_IMAGE_INPUTS", "1") == "1")
    parser.add_argument("--max-concurrent", type=int, default=int(_env("LLAMA_MAX_CONCURRENCY", str(DEFAULT_VLLM_MAX_CONCURRENCY))))
    parser.add_argument("--transport-mode", choices=["direct-endpoint", "pull-request"], default=_env("LLM_WORKER_TRANSPORT_MODE", "pull-request"))
    parser.add_argument("--test-mode", action="store_true", default=_env("LLM_WORKER_TEST_MODE", "0") == "1")
    parser.add_argument("--request-poll-seconds", type=float, default=float(_env("LLM_WORKER_REQUEST_POLL_SECONDS", "1.0")))
    args = parser.parse_args()

    router_url = _normalize_router_url(args.router_url or "")
    if not router_url:
        raise SystemExit("Set --router-url or LLM_ROUTER_URL.")

    model_target, model_name, local_model_root = _resolve_model_identity(
        test_mode=args.test_mode,
        model_path=args.model_path,
        model_dir=args.model_dir,
        model_file=args.model_file,
        model_name=args.model_name,
    )
    public_base_url = (args.public_base_url or f"http://127.0.0.1:{args.port}").strip().rstrip("/")

    child: subprocess.Popen[str] | None = None
    local_base_url = f"http://127.0.0.1:{args.port}"

    terminate = False
    worker_id: str | None = None
    state_lock = threading.Lock()
    needs_reregister = False

    def _handle_signal(_signum: int, _frame: object) -> None:
        nonlocal terminate
        terminate = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        if not args.test_mode:
            if model_target is None:
                raise SystemExit("No vLLM model target resolved.")
            command = _build_vllm_command(model_target, host=args.host, port=args.port)
            child = subprocess.Popen(command, env=os.environ.copy())
            _wait_for_local_server(local_base_url, args.startup_timeout_seconds)

        with httpx.Client(base_url=router_url, headers=_router_headers(), timeout=30.0) as client:
            def _register_worker(preferred_worker_id: str | None) -> str:
                register_response = client.post(
                    "/api/v1/workers/register",
                    json={
                        "worker_id": preferred_worker_id,
                        "display_name": args.display_name,
                        "hostname": socket.gethostname(),
                        "endpoint_base_url": public_base_url if args.transport_mode == "direct-endpoint" else None,
                        "model": model_name,
                        "local_model_path": model_target,
                        "local_model_root": local_model_root,
                        "max_concurrent_requests": args.max_concurrent,
                        "supports_image_inputs": args.supports_image_inputs,
                        "transport_mode": args.transport_mode,
                        "metadata": {
                            "launcher": "vllm_worker",
                            "runtime": "vllm",
                            "transport_mode": args.transport_mode,
                            "test_mode": args.test_mode,
                        },
                    },
                )
                register_response.raise_for_status()
                worker = register_response.json()["worker"]
                return str(worker["worker_id"])

            register_attempt = 0
            while not terminate:
                try:
                    worker_id = _register_worker(None)
                    break
                except Exception as exc:  # noqa: BLE001
                    if not _is_retryable_router_error(exc):
                        raise
                    delay = _router_retry_delay_seconds(attempt=register_attempt)
                    register_attempt += 1
                    _sleep_until(time.monotonic() + delay, terminate_flag=lambda: terminate)
            if worker_id is None:
                raise SystemExit("worker terminated before registration completed")

            def _heartbeat_loop() -> None:
                nonlocal worker_id, needs_reregister
                heartbeat_failures = 0
                while not terminate:
                    with state_lock:
                        active_worker_id = worker_id
                    try:
                        response = client.post(
                            f"/api/v1/workers/{active_worker_id}/heartbeat",
                            json={
                                "endpoint_base_url": public_base_url if args.transport_mode == "direct-endpoint" else None,
                                "model": model_name,
                                "local_model_path": model_target,
                                "local_model_root": local_model_root,
                                "max_concurrent_requests": args.max_concurrent,
                                "supports_image_inputs": args.supports_image_inputs,
                                "transport_mode": args.transport_mode,
                                "metadata": {
                                    "launcher": "vllm_worker",
                                    "runtime": "vllm",
                                    "transport_mode": args.transport_mode,
                                    "test_mode": args.test_mode,
                                },
                            },
                        )
                        response.raise_for_status()
                        heartbeat_failures = 0
                    except Exception as exc:  # noqa: BLE001
                        if _is_unknown_worker_error(exc):
                            with state_lock:
                                needs_reregister = True
                            _sleep_until(time.monotonic() + 1.0, terminate_flag=lambda: terminate)
                            continue
                        if _is_retryable_router_error(exc):
                            delay = _router_retry_delay_seconds(attempt=heartbeat_failures)
                            heartbeat_failures += 1
                            _sleep_until(time.monotonic() + delay, terminate_flag=lambda: terminate)
                            continue
                        return
                    for _ in range(max(1, int(args.heartbeat_seconds * 10))):
                        if terminate:
                            break
                        time.sleep(0.1)

            heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
            heartbeat_thread.start()
            pending_delivery: dict[str, Any] | None = None
            poll_failures = 0

            while not terminate:
                if child is not None and child.poll() is not None:
                    raise SystemExit(f"vLLM server exited with code {child.returncode}")
                with state_lock:
                    active_worker_id = worker_id
                    should_reregister = needs_reregister
                    if should_reregister:
                        needs_reregister = False
                if should_reregister:
                    try:
                        replacement_worker_id = _register_worker(active_worker_id)
                        with state_lock:
                            worker_id = replacement_worker_id
                    except Exception as exc:  # noqa: BLE001
                        if not _is_retryable_router_error(exc):
                            raise
                        _sleep_until(
                            time.monotonic() + _router_retry_delay_seconds(attempt=poll_failures),
                            terminate_flag=lambda: terminate,
                        )
                        poll_failures += 1
                    continue
                if pending_delivery is not None:
                    try:
                        delivery_response = client.post(
                            pending_delivery["path"],
                            json=pending_delivery["body"],
                        )
                        delivery_response.raise_for_status()
                        pending_delivery = None
                        poll_failures = 0
                    except Exception as exc:  # noqa: BLE001
                        if _is_unknown_worker_error(exc):
                            pending_delivery = None
                            with state_lock:
                                needs_reregister = True
                            continue
                        if _is_retryable_router_error(exc):
                            _sleep_until(
                                time.monotonic() + _router_retry_delay_seconds(attempt=poll_failures),
                                terminate_flag=lambda: terminate,
                            )
                            poll_failures += 1
                            continue
                        raise
                    continue
                try:
                    poll_response = client.post(
                        f"/api/v1/workers/{active_worker_id}/requests/poll",
                        json={"max_requests": 1},
                    )
                    poll_response.raise_for_status()
                    poll_failures = 0
                except Exception as exc:  # noqa: BLE001
                    if _is_unknown_worker_error(exc):
                        with state_lock:
                            needs_reregister = True
                        continue
                    if _is_retryable_router_error(exc):
                        _sleep_until(
                            time.monotonic() + _router_retry_delay_seconds(attempt=poll_failures),
                            terminate_flag=lambda: terminate,
                        )
                        poll_failures += 1
                        continue
                    raise
                requests = poll_response.json().get("requests") or []
                if not requests:
                    time.sleep(max(0.1, args.request_poll_seconds))
                    continue
                for item in requests:
                    request_id = str(item.get("request_id") or "").strip()
                    payload = item.get("payload")
                    if not request_id or not isinstance(payload, dict):
                        continue
                    try:
                        if args.test_mode:
                            response_body = _mock_chat_completion(payload, model_name=model_name)
                        else:
                            upstream_response = client.post(
                                local_base_url + "/v1/chat/completions",
                                json=payload,
                                headers={"Content-Type": "application/json"},
                                timeout=300.0,
                            )
                            upstream_response.raise_for_status()
                            response_body = upstream_response.json()
                            if not isinstance(response_body, dict):
                                raise RuntimeError("Local vLLM server did not return a JSON object")
                        pending_delivery = {
                            "path": f"/api/v1/workers/{active_worker_id}/requests/{request_id}/complete",
                            "body": {"response_body": response_body},
                        }
                    except Exception as exc:  # noqa: BLE001
                        error_text = str(exc)
                        error_status_code = None
                        if isinstance(exc, httpx.HTTPStatusError):
                            error_status_code = exc.response.status_code
                            response_text = exc.response.text.strip()
                            if response_text:
                                error_text = response_text
                        pending_delivery = {
                            "path": f"/api/v1/workers/{active_worker_id}/requests/{request_id}/fail",
                            "body": {"error": error_text, "status_code": error_status_code},
                        }
            heartbeat_thread.join(timeout=2.0)
            if worker_id is not None:
                try:
                    client.post(
                        f"/api/v1/workers/{worker_id}/disconnect",
                        json={"reason": "Worker shutting down."},
                    )
                except Exception:
                    pass
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
