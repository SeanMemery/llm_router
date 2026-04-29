from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


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
    result = [item for item in values if item and item.lower() not in {"none", "void"}]
    return result


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
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
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


def _download_target_path() -> Path | None:
    download_url = _env("LLAMA_MODEL_DOWNLOAD_URL")
    if not download_url:
        return None
    model_dir = Path(_env("LLAMA_MODEL_DIR", "/models")).expanduser()
    configured_name = _env("LLAMA_MODEL_DOWNLOAD_FILE") or _env("LLAMA_MODEL_FILE")
    if configured_name:
        filename = configured_name
    else:
        parsed = urlparse(download_url)
        filename = Path(parsed.path).name
    if not filename:
        raise SystemExit("Could not derive model filename from LLAMA_MODEL_DOWNLOAD_URL. Set LLAMA_MODEL_DOWNLOAD_FILE.")
    target = model_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return target

    command = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "5",
        "--continue-at",
        "-",
        "--output",
        str(target),
    ]
    bearer = _env("LLAMA_MODEL_DOWNLOAD_BEARER_TOKEN") or _env("HUGGING_FACE_HUB_TOKEN")
    if bearer:
        command.extend(["--header", f"Authorization: Bearer {bearer}"])
    extra_headers = _env("LLAMA_MODEL_DOWNLOAD_HEADERS")
    if extra_headers:
        for item in shlex.split(extra_headers):
            command.extend(["--header", item])
    command.append(download_url)
    subprocess.run(command, check=True)
    if not target.is_file() or target.stat().st_size <= 0:
        raise SystemExit(f"Downloaded model file is missing or empty: {target}")
    return target


def _resolve_model_path() -> Path:
    download_target = _download_target_path()
    if download_target is not None:
        return download_target
    model_path = _env("LLAMA_MODEL_PATH")
    if model_path:
        resolved = Path(model_path).expanduser()
        if not resolved.is_file():
            raise SystemExit(f"Configured LLAMA_MODEL_PATH does not exist: {resolved}")
        return resolved
    model_dir = _env("LLAMA_MODEL_DIR")
    if not model_dir:
        raise SystemExit("Set LLAMA_MODEL_DOWNLOAD_URL, LLAMA_MODEL_PATH, or LLAMA_MODEL_DIR.")
    root = Path(model_dir).expanduser()
    if not root.is_dir():
        raise SystemExit(f"Configured LLAMA_MODEL_DIR does not exist: {root}")
    configured_file = _env("LLAMA_MODEL_FILE")
    if configured_file:
        candidate = root / configured_file
        if not candidate.is_file():
            raise SystemExit(f"Configured LLAMA_MODEL_FILE does not exist under LLAMA_MODEL_DIR: {candidate}")
        return candidate
    candidates = [
        item
        for item in sorted(root.rglob("*.gguf"))
        if "mmproj" not in item.name.lower() and "projector" not in item.name.lower()
    ]
    if not candidates:
        raise SystemExit(f"No runnable GGUF files found under {root}")
    if len(candidates) > 1:
        raise SystemExit(
            "Multiple GGUF files found. Set LLAMA_MODEL_FILE or LLAMA_MODEL_PATH explicitly: "
            + ", ".join(str(item.relative_to(root)) for item in candidates[:8])
        )
    return candidates[0]


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
        "/opt/llm-router-worker/llama_worker.py",
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
    model_path = _resolve_model_path()
    base_port = _int_env("LLAMA_PORT", 8080)
    parallel_per_gpu = _int_env("LLM_WORKER_PARALLEL_PER_GPU", 2)
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
        child_env["LLAMA_MODEL_PATH"] = str(model_path)
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
