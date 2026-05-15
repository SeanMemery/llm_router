#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/opt/llm-router-worker:${PYTHONPATH:-}"
export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-/app}"
export LD_LIBRARY_PATH="${LLAMA_CPP_ROOT}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
PYTHON_BIN="${LLM_ROUTER_PYTHON_BIN:-$(command -v python3 || command -v python)}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
RUNTIME="${LLM_WORKER_RUNTIME:-llama-cpp}"
if [[ "${LLM_WORKER_MULTI_GPU:-0}" == "1" ]]; then
  if [[ "${RUNTIME}" == "vllm" ]]; then
    exec "${PYTHON_BIN}" /opt/llm-router-worker/worker_multi_gpu_vllm.py "$@"
  fi
  exec "${PYTHON_BIN}" /opt/llm-router-worker/worker_multi_gpu.py "$@"
fi
if [[ "${RUNTIME}" == "vllm" ]]; then
  exec "${PYTHON_BIN}" /opt/llm-router-worker/vllm_worker.py "$@"
fi
exec "${PYTHON_BIN}" /opt/llm-router-worker/llama_worker.py "$@"
