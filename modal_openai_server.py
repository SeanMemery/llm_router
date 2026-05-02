from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "RedHatAI/Qwen3.6-35B-A3B-NVFP4"
DEFAULT_GPU = "RTX4090"
DEFAULT_SERVED_MODEL_NAME = "llm"
DEFAULT_PYTHON_VERSION = "3.11"
DEFAULT_VLLM_VERSION = "0.19.1"
VLLM_PORT = 8000
DEFAULT_UV_HTTP_TIMEOUT_SECONDS = "600"
DEFAULT_UV_HTTP_RETRIES = "8"


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    text = value.strip()
    return text or default


def _sleep_until(deadline_seconds: float, *, terminate_flag: callable) -> None:
    while not terminate_flag():
        remaining = deadline_seconds - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.25, remaining))


def _build_modal_app(
    *,
    model: str,
    served_model_name: str,
    gpu: str,
    max_model_len: int,
    api_key: str | None,
    app_name: str,
):
    import modal

    image = (
        modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python=DEFAULT_PYTHON_VERSION)
        .entrypoint([])
        .env(
            {
                "UV_HTTP_TIMEOUT": DEFAULT_UV_HTTP_TIMEOUT_SECONDS,
                "UV_HTTP_RETRIES": DEFAULT_UV_HTTP_RETRIES,
                "PIP_DEFAULT_TIMEOUT": DEFAULT_UV_HTTP_TIMEOUT_SECONDS,
                "HF_XET_HIGH_PERFORMANCE": "1",
            }
        )
        .uv_pip_install(f"vllm=={DEFAULT_VLLM_VERSION}")
    )
    hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
    vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

    app = modal.App(app_name)

    @app.function(
        image=image,
        gpu=gpu,
        scaledown_window=15 * 60,
        timeout=10 * 60,
        serialized=True,
        volumes={
            "/root/.cache/huggingface": hf_cache_vol,
            "/root/.cache/vllm": vllm_cache_vol,
        },
    )
    @modal.concurrent(max_inputs=64)
    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
    def serve() -> None:
        cmd = [
            "vllm",
            "serve",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--served-model-name",
            served_model_name,
            "--uvicorn-log-level",
            "info",
            "--async-scheduling",
            "--enforce-eager",
            "--max-model-len",
            str(max_model_len),
            "--trust-remote-code",
            "--limit-mm-per-prompt",
            json.dumps({"image": 0, "video": 0, "audio": 0}),
        ]
        if api_key:
            cmd.extend(["--api-key", api_key])
        subprocess.Popen(cmd)

    return app, serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch a managed Modal OpenAI-compatible vLLM endpoint.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--duration-minutes", type=int, default=120)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--app-name", default="llm-router-modal-openai")
    parser.add_argument("--launch-id", default="")
    parser.add_argument("--state-path", required=True)
    args = parser.parse_args(argv)

    state_path = Path(args.state_path).resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    launch_id = (args.launch_id or f"modal-{int(time.time())}").strip()
    deadline_seconds = time.monotonic() + max(60, int(args.duration_minutes) * 60)
    api_key = args.api_key.strip() or _env("PATTERN_LLM_API_KEY") or _env("OPENAI_API_KEY")

    should_terminate = False

    def _request_shutdown(signum: int, _frame: Any) -> None:
        nonlocal should_terminate
        should_terminate = True
        print(f"[modal-openai] signal={signum} requested shutdown", flush=True)

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    _write_state(
        state_path,
        {
            "launch_id": launch_id,
            "state": "starting",
            "app_name": args.app_name.strip(),
            "model": args.model.strip(),
            "served_model_name": args.served_model_name.strip(),
            "gpu": args.gpu.strip(),
            "max_model_len": int(max(1024, args.max_model_len)),
            "duration_minutes": int(max(1, args.duration_minutes)),
            "api_key_set": bool(api_key),
            "started_at_epoch": int(time.time()),
            "last_error": None,
        },
    )

    app_id: str | None = None
    web_url: str | None = None
    modal = None
    app = None
    final_state = "stopped"
    final_error: str | None = None
    try:
        try:
            import modal as modal_module
        except ModuleNotFoundError as exc:
            raise RuntimeError("Modal package not installed in llm_router environment.") from exc

        modal = modal_module
        app, serve = _build_modal_app(
            model=args.model.strip(),
            served_model_name=args.served_model_name.strip(),
            gpu=args.gpu.strip(),
            max_model_len=int(max(1024, args.max_model_len)),
            api_key=api_key,
            app_name=args.app_name.strip(),
        )

        with modal.enable_output():
            with app.run():
                app_id = app.app_id
                web_url = serve.get_web_url()
                _write_state(
                    state_path,
                    {
                        "launch_id": launch_id,
                        "state": "running",
                        "app_name": args.app_name.strip(),
                        "app_id": app_id,
                        "web_url": web_url,
                        "model": args.model.strip(),
                        "served_model_name": args.served_model_name.strip(),
                        "gpu": args.gpu.strip(),
                        "max_model_len": int(max(1024, args.max_model_len)),
                        "duration_minutes": int(max(1, args.duration_minutes)),
                        "api_key_set": bool(api_key),
                        "started_at_epoch": int(time.time()),
                        "last_error": None,
                    },
                )
                print(
                    f"[modal-openai] launch_id={launch_id} app_id={app_id} web_url={web_url}",
                    flush=True,
                )
                _sleep_until(deadline_seconds, terminate_flag=lambda: should_terminate)
    except Exception as exc:  # noqa: BLE001
        final_state = "failed"
        final_error = str(exc).strip() or exc.__class__.__name__
        _write_state(
            state_path,
            {
                "launch_id": launch_id,
                "state": "failed",
                "app_name": args.app_name.strip(),
                "app_id": app_id,
                "web_url": web_url,
                "model": args.model.strip(),
                "served_model_name": args.served_model_name.strip(),
                "gpu": args.gpu.strip(),
                "max_model_len": int(max(1024, args.max_model_len)),
                "duration_minutes": int(max(1, args.duration_minutes)),
                "api_key_set": bool(api_key),
                "started_at_epoch": int(time.time()),
                "last_error": final_error,
            },
        )
        raise
    finally:
        if app_id:
            try:
                subprocess.run(
                    [sys.executable, "-m", "modal", "app", "stop", "-y", app_id],
                    check=False,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    text=True,
                )
            except Exception:
                pass
        _write_state(
            state_path,
            {
                "launch_id": launch_id,
                "state": final_state,
                "app_name": args.app_name.strip(),
                "app_id": app_id,
                "web_url": web_url,
                "model": args.model.strip(),
                "served_model_name": args.served_model_name.strip(),
                "gpu": args.gpu.strip(),
                "max_model_len": int(max(1024, args.max_model_len)),
                "duration_minutes": int(max(1, args.duration_minutes)),
                "api_key_set": bool(api_key),
                "started_at_epoch": int(time.time()),
                "last_error": final_error,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
