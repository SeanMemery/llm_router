import argparse
import concurrent.futures
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

try:
    from env_utils import load_dotenv_file
except ModuleNotFoundError:
    def load_dotenv_file(path: Path) -> dict[str, str]:
        loaded: dict[str, str] = {}
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return loaded
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if not key:
                continue
            os.environ.setdefault(key, value)
            loaded[key] = value
        return loaded


DEFAULT_MODAL_MODEL = "RedHatAI/Qwen3.6-35B-A3B-NVFP4"
DEFAULT_MODAL_GPU = "A100-40GB"
DEFAULT_MODAL_MAX_MODEL_LEN = 8192
DEFAULT_MODAL_MAX_TOKENS = 8192
DEFAULT_MODAL_EXECUTION_MODE = "vllm"
DEFAULT_MODAL_PYTHON_VERSION = "3.11"
DEFAULT_MODAL_VLLM_VERSION = "0.19.1"
DEFAULT_MODAL_TRANSFORMERS_SPEC = "git+https://github.com/huggingface/transformers.git"


def _sleep_until(deadline_seconds: float, *, terminate_flag: callable) -> None:
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


def _require_modal() -> Any:
    spec = importlib.util.find_spec("modal")
    if spec is None:
        raise SystemExit(
            "Modal is not installed in this environment. Install it with `uv add modal` or `pip install modal`."
        )
    import modal

    return modal


def _write_status(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stop_modal_app(*, python_executable: str, app_id: str | None) -> None:
    if app_id is None or not app_id.strip():
        return
    try:
        subprocess.run(
            [python_executable, "-m", "modal", "app", "stop", "-y", app_id.strip()],
            check=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )
    except Exception:
        return


_MODAL_SPEC = importlib.util.find_spec("modal")
_MODAL: Any | None = None
if _MODAL_SPEC is not None:
    import modal as _modal

    _MODAL = _modal
    MODAL_TEST_IMAGE = _MODAL.Image.debian_slim(python_version=DEFAULT_MODAL_PYTHON_VERSION)
    MODAL_VLLM_IMAGE = (
        _MODAL.Image.from_registry("vllm/vllm-openai:latest", add_python=DEFAULT_MODAL_PYTHON_VERSION)
        .apt_install("git")
        .entrypoint([])
        .run_commands(
            "uv pip install --system --upgrade "
            f"huggingface-hub {DEFAULT_MODAL_TRANSFORMERS_SPEC} "
            f"vllm=={DEFAULT_MODAL_VLLM_VERSION}",
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    )
    MODAL_HF_CACHE = _MODAL.Volume.from_name("huggingface-cache", create_if_missing=True)
    MODAL_VLLM_CACHE = _MODAL.Volume.from_name("vllm-cache", create_if_missing=True)


def _build_modal_runtime(
    *,
    modal: Any,
    model_name: str,
    app_name: str,
    execution_mode: str,
    gpu_kind: str,
    max_model_len: int,
    default_max_tokens: int,
) -> tuple[Any, Any]:
    if _MODAL is None:
        raise SystemExit("Modal is not installed in this environment.")
    app = modal.App(app_name)

    @app.cls(
        image=MODAL_TEST_IMAGE,
        timeout=300,
        serialized=True,
    )
    class ModalTestServer:
        configured_model_name: str = modal.parameter()

        @modal.enter()
        def start(self) -> None:
            self._model_name = self.configured_model_name

        @modal.method()
        def ready(self) -> dict[str, Any]:
            return {
                "status": "ready",
                "model": self._model_name,
                "execution_mode": "test",
            }

        @modal.method()
        def process(self, payload: dict[str, Any]) -> dict[str, Any]:
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
            prompt_tokens = max(1, len(prompt_preview.split()) or 1)
            completion_text = f"modal test worker response: {prompt_preview}".strip()
            completion_tokens = max(1, len(completion_text.split()) or 1)
            return {
                "id": f"chatcmpl-modal-test-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self._model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": completion_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens),
                },
                "router": {
                    "worker_execution_mode": "modal-test",
                },
            }

    @app.cls(
        image=MODAL_VLLM_IMAGE,
        gpu=gpu_kind,
        volumes={
            "/root/.cache/huggingface": MODAL_HF_CACHE,
            "/root/.cache/vllm": MODAL_VLLM_CACHE,
        },
        timeout=600,
        serialized=True,
    )
    class ModalVllmServer:
        configured_model_name: str = modal.parameter()
        configured_max_model_len: int = modal.parameter()
        configured_default_max_tokens: int = modal.parameter()

        @modal.enter()
        def start(self) -> None:
            import vllm

            self._model_name = self.configured_model_name
            self._default_max_tokens = self.configured_default_max_tokens
            llm_kwargs: dict[str, Any] = {
                "model": self.configured_model_name,
                "max_model_len": self.configured_max_model_len,
                "attention_backend": "flashinfer",
                "async_scheduling": True,
            }
            if str(self.configured_model_name).strip() == "RedHatAI/Qwen3.6-35B-A3B-NVFP4":
                llm_kwargs["moe_backend"] = "marlin"
            self._llm = vllm.LLM(**llm_kwargs)
            self._sampling_params = self._llm.get_default_sampling_params()
            self._sampling_params.temperature = 0.0
            self._sampling_params.max_tokens = self._default_max_tokens
            tokenizer = None
            try:
                tokenizer = self._llm.get_tokenizer()
            except Exception:
                tokenizer = None
            self._tokenizer = tokenizer

        @modal.method()
        def ready(self) -> dict[str, Any]:
            return {
                "status": "ready",
                "model": self._model_name,
                "execution_mode": "vllm",
            }

        def _content_to_text(self, content: Any) -> str:
            if isinstance(content, str):
                return content
            if not isinstance(content, list):
                return ""
            fragments: list[str] = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        fragments.append(item.strip())
                    continue
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "").strip().lower()
                if item_type and item_type not in {"text", "input_text"}:
                    continue
                for key in ("text", "content", "value", "input_text"):
                    candidate = item.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        fragments.append(candidate.strip())
                        break
            return "\n\n".join(fragments)

        def _normalize_messages(self, payload: dict[str, Any]) -> list[dict[str, str]]:
            messages = payload.get("messages")
            if not isinstance(messages, list):
                return [{"role": "user", "content": ""}]
            normalized: list[dict[str, str]] = []
            for item in messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user").strip() or "user"
                content = self._content_to_text(item.get("content"))
                normalized.append({"role": role, "content": content})
            return normalized or [{"role": "user", "content": ""}]

        def _count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
            tokenizer = getattr(self, "_tokenizer", None)
            if tokenizer is None:
                content = "\n".join(item.get("content", "") for item in messages)
                return max(1, len(content.split()) or 1)
            try:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                return max(1, len(token_ids))
            except Exception:
                content = "\n".join(item.get("content", "") for item in messages)
                return max(1, len(content.split()) or 1)

        def _count_completion_tokens(self, text: str) -> int:
            tokenizer = getattr(self, "_tokenizer", None)
            if tokenizer is None:
                return max(1, len(text.split()) or 1)
            try:
                return max(1, len(tokenizer.encode(text)))
            except Exception:
                return max(1, len(text.split()) or 1)

        @modal.method()
        def process(self, payload: dict[str, Any]) -> dict[str, Any]:
            messages = self._normalize_messages(payload)
            sampling_params = self._sampling_params.clone()
            temperature = payload.get("temperature")
            if isinstance(temperature, (int, float)):
                sampling_params.temperature = float(temperature)
            max_tokens = payload.get("max_tokens")
            if isinstance(max_tokens, int) and max_tokens > 0:
                sampling_params.max_tokens = max_tokens
            elif isinstance(max_tokens, float) and max_tokens > 0:
                sampling_params.max_tokens = int(max_tokens)
            else:
                sampling_params.max_tokens = self._default_max_tokens
            top_p = payload.get("top_p")
            if isinstance(top_p, (int, float)) and 0.0 < float(top_p) <= 1.0:
                sampling_params.top_p = float(top_p)
            stop = payload.get("stop")
            if isinstance(stop, str) and stop.strip():
                sampling_params.stop = [stop]
            elif isinstance(stop, list):
                sampling_params.stop = [str(item) for item in stop if str(item).strip()]

            outputs = self._llm.chat([messages], sampling_params=sampling_params)
            output = outputs[0].outputs[0]
            text = str(output.text or "")
            prompt_tokens = self._count_prompt_tokens(messages)
            completion_tokens = self._count_completion_tokens(text)
            finish_reason = "stop"
            stop_reason = getattr(output, "stop_reason", None)
            if isinstance(stop_reason, str) and stop_reason.strip():
                finish_reason = stop_reason.strip()
            return {
                "id": f"chatcmpl-modal-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self._model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens),
                },
                "router": {
                    "worker_execution_mode": "modal",
                    "modal_gpu": gpu_kind,
                },
            }

    if execution_mode == "test":
        return app, ModalTestServer(configured_model_name=model_name)
    return app, ModalVllmServer(
        configured_model_name=model_name,
        configured_max_model_len=max_model_len,
        configured_default_max_tokens=default_max_tokens,
    )


def main() -> int:
    script_root = Path(__file__).resolve().parent
    load_dotenv_file(script_root / ".env")

    parser = argparse.ArgumentParser(description="Modal-backed pull-request worker for llm_router")
    parser.add_argument("--router-url", default=_env("LLM_ROUTER_URL"))
    parser.add_argument("--model", default=_env("MODAL_LLM_MODEL", DEFAULT_MODAL_MODEL))
    parser.add_argument("--display-name", default=_env("MODAL_WORKER_DISPLAY_NAME") or socket.gethostname())
    parser.add_argument("--heartbeat-seconds", type=float, default=float(_env("LLM_WORKER_HEARTBEAT_SECONDS", "15")))
    parser.add_argument("--request-poll-seconds", type=float, default=float(_env("LLM_WORKER_REQUEST_POLL_SECONDS", "0.5")))
    parser.add_argument("--max-concurrent", type=int, default=int(_env("MODAL_WORKER_MAX_CONCURRENT", "8")))
    parser.add_argument("--duration-minutes", type=int, default=int(_env("MODAL_WORKER_DURATION_MINUTES", "60")))
    parser.add_argument("--launch-id", default=_env("MODAL_WORKER_LAUNCH_ID"))
    parser.add_argument("--gpu", default=_env("MODAL_WORKER_GPU", DEFAULT_MODAL_GPU))
    parser.add_argument("--max-model-len", type=int, default=int(_env("MODAL_WORKER_MAX_MODEL_LEN", str(DEFAULT_MODAL_MAX_MODEL_LEN))))
    parser.add_argument("--max-tokens", type=int, default=int(_env("MODAL_WORKER_MAX_TOKENS", str(DEFAULT_MODAL_MAX_TOKENS))))
    parser.add_argument("--execution-mode", choices=["vllm", "test"], default=_env("MODAL_WORKER_EXECUTION_MODE", DEFAULT_MODAL_EXECUTION_MODE))
    parser.add_argument("--state-path", default=_env("MODAL_WORKER_STATE_PATH"))
    args = parser.parse_args()

    router_url = (args.router_url or "").strip().rstrip("/")
    if not router_url:
        raise SystemExit("Set --router-url or LLM_ROUTER_URL.")
    model_name = (args.model or "").strip()
    if not model_name:
        raise SystemExit("Set --model or MODAL_LLM_MODEL.")
    display_name = (args.display_name or "").strip()
    if not display_name:
        raise SystemExit("Set --display-name or MODAL_WORKER_DISPLAY_NAME.")
    duration_minutes = max(1, int(args.duration_minutes))
    max_concurrent = max(1, int(args.max_concurrent))

    modal = _require_modal()
    launch_id = (args.launch_id or f"modal-{int(time.time())}").strip()
    app_name = f"llm-router-{launch_id}"
    state_path = Path(args.state_path).expanduser() if args.state_path else None
    app, modal_server = _build_modal_runtime(
        modal=modal,
        model_name=model_name,
        app_name=app_name,
        execution_mode=args.execution_mode,
        gpu_kind=args.gpu,
        max_model_len=max(1024, int(args.max_model_len)),
        default_max_tokens=max(1, int(args.max_tokens)),
    )

    terminate = False
    worker_id: str | None = None
    needs_reregister = False
    fatal_error_message: str | None = None
    worker_id_lock = threading.Lock()
    inflight_lock = threading.Lock()
    inflight_count = 0
    deadline = datetime.now(UTC) + timedelta(minutes=duration_minutes)
    modal_app_id: str | None = None

    def _handle_signal(_signum: int, _frame: object) -> None:
        nonlocal terminate
        terminate = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        with modal.enable_output():
            with app.run():
                modal_app_id = app.app_id
                print(
                    f"[modal-worker] launch_id={launch_id} app_name={app_name} app_id={modal_app_id} mode={args.execution_mode}",
                    flush=True,
                )
                _write_status(
                    state_path,
                    {
                        "launch_id": launch_id,
                        "app_name": app_name,
                        "app_id": modal_app_id,
                        "execution_mode": args.execution_mode,
                        "state": "modal_running",
                    },
                )
                server = modal_server
                _write_status(
                    state_path,
                    {
                        "launch_id": launch_id,
                        "app_name": app_name,
                        "app_id": modal_app_id,
                        "execution_mode": args.execution_mode,
                        "state": "warming_up",
                    },
                )
                ready_payload = server.ready.remote()
                print(
                    f"[modal-worker] launch_id={launch_id} ready={ready_payload}",
                    flush=True,
                )
                with httpx.Client(base_url=router_url, headers=_router_headers(), timeout=60.0) as client:
                    def _record_failed_state(error_message: str) -> None:
                        _write_status(
                            state_path,
                            {
                                "launch_id": launch_id,
                                "app_name": app_name,
                                "app_id": modal_app_id,
                                "execution_mode": args.execution_mode,
                                "state": "failed",
                                "worker_id": worker_id,
                                "last_error": error_message,
                            },
                        )

                    def _request_fatal_shutdown(error_message: str) -> None:
                        nonlocal terminate, fatal_error_message
                        if fatal_error_message is None:
                            fatal_error_message = error_message
                        _record_failed_state(error_message)
                        current_worker_id = worker_id
                        if current_worker_id:
                            try:
                                client.post(
                                    f"/api/v1/workers/{current_worker_id}/disconnect",
                                    json={"reason": error_message[:400]},
                                ).raise_for_status()
                            except Exception:
                                pass
                        terminate = True

                    def _register_worker(preferred_worker_id: str | None) -> str:
                        response = client.post(
                            "/api/v1/workers/register",
                            json={
                                "worker_id": preferred_worker_id,
                                "display_name": display_name,
                                "hostname": socket.gethostname(),
                                "endpoint_base_url": None,
                                "model": model_name,
                                "local_model_path": None,
                                "local_model_root": None,
                                "max_concurrent_requests": max_concurrent,
                                "supports_image_inputs": False,
                                "transport_mode": "pull-request",
                                "metadata": {
                                    "launcher": "modal_worker",
                                    "launch_id": launch_id,
                                    "modal_app_name": app_name,
                                    "modal_app_id": modal_app_id,
                                    "modal_gpu": args.gpu,
                                    "duration_minutes": duration_minutes,
                                    "desired_router_priority": 3,
                                    "execution_mode": args.execution_mode,
                                },
                            },
                        )
                        response.raise_for_status()
                        return str(response.json()["worker"]["worker_id"])

                    register_attempt = 0
                    while not terminate:
                        try:
                            worker_id = _register_worker(None)
                            _write_status(
                                state_path,
                                {
                                    "launch_id": launch_id,
                                    "app_name": app_name,
                                    "app_id": modal_app_id,
                                    "execution_mode": args.execution_mode,
                                    "state": "registered",
                                    "worker_id": worker_id,
                                },
                            )
                            break
                        except Exception as exc:  # noqa: BLE001
                            print(f"[modal-worker] register retry after error: {exc}", flush=True)
                            if not _is_retryable_router_error(exc):
                                raise
                            delay = _router_retry_delay_seconds(attempt=register_attempt)
                            register_attempt += 1
                            _sleep_until(time.monotonic() + delay, terminate_flag=lambda: terminate)
                    if worker_id is None:
                        raise SystemExit("Worker terminated before router registration completed.")

                    def _heartbeat_loop() -> None:
                        nonlocal worker_id, needs_reregister
                        heartbeat_attempt = 0
                        while not terminate:
                            with worker_id_lock:
                                current_worker_id = worker_id
                            if current_worker_id is None:
                                return
                            payload = {
                                "model": model_name,
                                "max_concurrent_requests": max_concurrent,
                                "supports_image_inputs": False,
                                "transport_mode": "pull-request",
                                "metadata": {
                                    "launcher": "modal_worker",
                                    "launch_id": launch_id,
                                    "modal_app_name": app_name,
                                    "modal_app_id": modal_app_id,
                                    "modal_gpu": args.gpu,
                                    "duration_minutes": duration_minutes,
                                    "expires_at": deadline.isoformat().replace("+00:00", "Z"),
                                    "desired_router_priority": 3,
                                    "execution_mode": args.execution_mode,
                                },
                            }
                            try:
                                client.post(
                                    f"/api/v1/workers/{current_worker_id}/heartbeat",
                                    json=payload,
                                ).raise_for_status()
                                heartbeat_attempt = 0
                            except Exception as exc:  # noqa: BLE001
                                if _is_unknown_worker_error(exc):
                                    with worker_id_lock:
                                        needs_reregister = True
                                    return
                                if _is_retryable_router_error(exc):
                                    delay = _router_retry_delay_seconds(attempt=heartbeat_attempt)
                                    heartbeat_attempt += 1
                                    _sleep_until(time.monotonic() + delay, terminate_flag=lambda: terminate)
                                    continue
                            _sleep_until(time.monotonic() + args.heartbeat_seconds, terminate_flag=lambda: terminate)

                    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
                    heartbeat_thread.start()

                    def _process_request(request_payload: dict[str, Any]) -> None:
                        nonlocal worker_id, needs_reregister, inflight_count
                        request_id = str(request_payload["request_id"])
                        payload = request_payload["payload"]
                        try:
                            print(f"[modal-worker] processing request {request_id}", flush=True)
                            response_body = server.process.remote(payload)
                            client.post(
                                f"/api/v1/workers/{worker_id}/requests/{request_id}/complete",
                                json={"response_body": response_body},
                            ).raise_for_status()
                        except Exception as exc:  # noqa: BLE001
                            if _is_unknown_worker_error(exc):
                                with worker_id_lock:
                                    needs_reregister = True
                                return
                            error_message = str(exc).strip() or exc.__class__.__name__
                            status_code = 502 if _is_retryable_router_error(exc) else 500
                            try:
                                client.post(
                                    f"/api/v1/workers/{worker_id}/requests/{request_id}/fail",
                                    json={"error": error_message, "status_code": status_code},
                                ).raise_for_status()
                            except Exception:
                                pass
                            if status_code >= 500 and not _is_retryable_router_error(exc):
                                _request_fatal_shutdown(f"modal worker fatal error: {error_message}")
                        finally:
                            with inflight_lock:
                                inflight_count = max(0, inflight_count - 1)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
                        while not terminate:
                            if datetime.now(UTC) >= deadline:
                                terminate = True
                                break
                            with worker_id_lock:
                                if needs_reregister:
                                    register_attempt = 0
                                    while not terminate:
                                        try:
                                            worker_id = _register_worker(worker_id)
                                            needs_reregister = False
                                            break
                                        except Exception as exc:  # noqa: BLE001
                                            print(f"[modal-worker] reregister retry after error: {exc}", flush=True)
                                            if not _is_retryable_router_error(exc):
                                                raise
                                            delay = _router_retry_delay_seconds(attempt=register_attempt)
                                            register_attempt += 1
                                            _sleep_until(time.monotonic() + delay, terminate_flag=lambda: terminate)
                                    if terminate:
                                        break
                            with inflight_lock:
                                available_slots = max_concurrent - inflight_count
                            if available_slots <= 0:
                                time.sleep(0.05)
                                continue
                            try:
                                poll = client.post(
                                    f"/api/v1/workers/{worker_id}/requests/poll",
                                    json={"max_requests": available_slots},
                                )
                                poll.raise_for_status()
                                requests = poll.json().get("requests", [])
                            except Exception as exc:  # noqa: BLE001
                                if _is_unknown_worker_error(exc):
                                    with worker_id_lock:
                                        needs_reregister = True
                                    continue
                                if _is_retryable_router_error(exc):
                                    time.sleep(args.request_poll_seconds)
                                    continue
                                raise
                            if not requests:
                                time.sleep(args.request_poll_seconds)
                                continue
                            for request_payload in requests:
                                with inflight_lock:
                                    inflight_count += 1
                                pool.submit(_process_request, request_payload)

                    heartbeat_thread.join(timeout=2.0)
                    if worker_id is not None:
                        try:
                            client.post(
                                f"/api/v1/workers/{worker_id}/disconnect",
                                json={
                                    "reason": (
                                        fatal_error_message
                                        or ("expired" if datetime.now(UTC) >= deadline else "shutdown")
                                    )[:400]
                                },
                            )
                        except Exception:
                            pass
    finally:
        _write_status(
            state_path,
            {
                "launch_id": launch_id,
                "app_name": app_name,
                "app_id": modal_app_id,
                "execution_mode": args.execution_mode,
                "state": "failed" if fatal_error_message else "stopped",
                "worker_id": worker_id,
                "last_error": fatal_error_message,
            },
        )
        _stop_modal_app(python_executable=sys.executable, app_id=modal_app_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
