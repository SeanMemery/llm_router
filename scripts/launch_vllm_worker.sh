#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export LLM_WORKER_RUNTIME="${LLM_WORKER_RUNTIME:-vllm}"

if [[ -z "${LLM_ROUTER_URL:-}" ]]; then
  echo "Missing LLM_ROUTER_URL" >&2
  exit 1
fi

if [[ -z "${LLAMA_MODEL_PATH:-}" && -z "${LLAMA_MODEL_DIR:-}" && -z "${LLAMA_MODEL_NAME:-}" ]]; then
  echo "Set one of LLAMA_MODEL_PATH, LLAMA_MODEL_DIR, or LLAMA_MODEL_NAME" >&2
  exit 1
fi

export LLAMA_PORT="${LLAMA_PORT:-8080}"
export LLAMA_MAX_CONCURRENCY="${LLAMA_MAX_CONCURRENCY:-12}"
export LLM_WORKER_TRANSPORT_MODE="${LLM_WORKER_TRANSPORT_MODE:-pull-request}"
export LLM_WORKER_HEARTBEAT_SECONDS="${LLM_WORKER_HEARTBEAT_SECONDS:-15}"
export LLM_WORKER_REQUEST_POLL_SECONDS="${LLM_WORKER_REQUEST_POLL_SECONDS:-1.0}"
export LLM_WORKER_STARTUP_TIMEOUT_SECONDS="${LLM_WORKER_STARTUP_TIMEOUT_SECONDS:-900}"
export LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"

if [[ -z "${LLAMA_DISPLAY_NAME:-}" ]]; then
  HOST_TAG="$(hostname | tr -cd '[:alnum:]-' | cut -c1-32)"
  export LLAMA_DISPLAY_NAME="vllm-${HOST_TAG:-worker}"
fi

PYTHON_BIN="${LLM_ROUTER_PYTHON_BIN:-python3}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/vllm_worker.py" "$@"
