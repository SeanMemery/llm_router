from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from vllm_worker import DEFAULT_VLLM_MAX_CONCURRENCY, _normalize_model_reference


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


def _require(name: str) -> str:
    value = _env(name)
    if value is None:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _int_env(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer value for {name}: {raw}") from exc
    if value < 1:
        raise SystemExit(f"{name} must be >= 1")
    return value


def _parse_gpu_indices(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",")]
    return [item for item in values if item and item.lower() not in {"none", "void"}]


def _detected_gpu_indices() -> list[str]:
    configured = _env("LLAMA_GPU_INDICES")
    if configured:
        indices = _parse_gpu_indices(configured)
        if indices:
            return indices
    nvidia_visible = _env("NVIDIA_VISIBLE_DEVICES")
    if nvidia_visible and nvidia_visible.lower() not in {"all"}:
        indices = _parse_gpu_indices(nvidia_visible)
        if indices:
            return indices
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Unable to detect GPUs with nvidia-smi. Set LLAMA_GPU_INDICES explicitly.") from exc
    indices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not indices:
        raise SystemExit("No GPUs detected for multi-GPU worker launch.")
    return indices


def _resolve_model_target() -> str:
    model_path = _normalize_model_reference(_env("LLAMA_MODEL_PATH"))
    if model_path:
        return model_path
    model_dir = _env("LLAMA_MODEL_DIR")
    if model_dir:
        root = Path(model_dir).expanduser()
        configured_file = _env("LLAMA_MODEL_FILE")
        if configured_file:
            candidate = root / configured_file
            if candidate.exists():
                return str(candidate)
        normalized_dir = _normalize_model_reference(model_dir)
        if normalized_dir:
            return normalized_dir
    model_name = _normalize_model_reference(_env("LLAMA_MODEL_NAME"))
    if model_name:
        return model_name
    raise SystemExit("Set LLAMA_MODEL_PATH, LLAMA_MODEL_DIR, or LLAMA_MODEL_NAME.")


def _public_base_url_for_gpu(*, template: str | None, gpu_index: str, port: int, worker_index: int) -> str | None:
    if not template:
        return None
    if "{" in template:
        return template.format(gpu=gpu_index, port=port, index=worker_index)
    if worker_index > 0:
        raise SystemExit(
            "LLM_WORKER_PUBLIC_BASE_URL must contain {port}, {gpu}, or {index} placeholders when launching multiple direct-endpoint workers."
        )
    return template.rstrip("/")


def _worker_command(*, port: int, display_name: str, public_base_url: str | None, max_concurrent: int) -> list[str]:
    command = [
        sys.executable,
        "/opt/llm-router-worker/vllm_worker.py",
        "--router-url",
        _require("LLM_ROUTER_URL"),
        "--port",
        str(port),
        "--display-name",
        display_name,
        "--max-concurrent",
        str(max_concurrent),
        "--transport-mode",
        _env("LLM_WORKER_TRANSPORT_MODE", "pull-request") or "pull-request",
    ]
    if public_base_url:
        command.extend(["--public-base-url", public_base_url])
    model_name = _env("LLAMA_MODEL_NAME")
    if model_name:
        command.extend(["--model-name", model_name])
    supports_images = _env("LLAMA_SUPPORTS_IMAGE_INPUTS", "0")
    if supports_images == "1":
        command.append("--supports-image-inputs")
    return command


def main() -> int:
    gpu_indices = _detected_gpu_indices()
    model_target = _resolve_model_target()
    base_port = _int_env("LLAMA_PORT", 8080)
    parallel_per_gpu = _int_env("LLM_WORKER_PARALLEL_PER_GPU", DEFAULT_VLLM_MAX_CONCURRENCY)
    display_name_base = _env("LLAMA_DISPLAY_NAME", socket.gethostname()) or socket.gethostname()
    public_base_url_template = _env("LLM_WORKER_PUBLIC_BASE_URL")

    children: list[subprocess.Popen[str]] = []
    terminating = False

    def _terminate_children() -> None:
        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic() + 20.0
        for child in children:
            remaining = max(0.0, deadline - time.monotonic())
            if child.poll() is None:
                try:
                    child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    child.kill()
        for child in children:
            if child.poll() is None:
                child.wait(timeout=5)

    def _handle_signal(_signum: int, _frame: object) -> None:
        nonlocal terminating
        terminating = True
        _terminate_children()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    for worker_index, gpu_index in enumerate(gpu_indices):
        port = base_port + worker_index
        public_base_url = _public_base_url_for_gpu(
            template=public_base_url_template,
            gpu_index=gpu_index,
            port=port,
            worker_index=worker_index,
        )
        display_name = f"{display_name_base}-gpu{gpu_index}"
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = gpu_index
        child_env["NVIDIA_VISIBLE_DEVICES"] = gpu_index
        child_env["HIP_VISIBLE_DEVICES"] = gpu_index
        child_env["ROCR_VISIBLE_DEVICES"] = gpu_index
        child_env["GPU_DEVICE_ORDINAL"] = gpu_index
        child_env["LLAMA_MODEL_PATH"] = model_target
        child_env["LLAMA_MAX_CONCURRENCY"] = str(parallel_per_gpu)
        if public_base_url:
            child_env["LLM_WORKER_PUBLIC_BASE_URL"] = public_base_url
        command = _worker_command(
            port=port,
            display_name=display_name,
            public_base_url=public_base_url,
            max_concurrent=parallel_per_gpu,
        )
        children.append(subprocess.Popen(command, env=child_env))

    try:
        while not terminating:
            for child in children:
                exit_code = child.poll()
                if exit_code is not None:
                    terminating = True
                    _terminate_children()
                    return exit_code
            time.sleep(1.0)
    finally:
        _terminate_children()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
