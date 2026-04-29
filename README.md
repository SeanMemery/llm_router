# llm_router

`llm_router` is an OpenAI-compatible router for mixing local workers and saved remote endpoints behind one API.

It exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`

It supports:

- saved URL backends
- self-registering worker backends
- pull-request workers
- a separate UI process for inspecting connections, requests, and worker state
- optional local Modal-launched workers from the UI

## Layout

- `app.py`: router backend
- `ui_app.py`: standalone UI server
- `llama_worker.py`: llama.cpp worker process
- `vllm_worker.py`: vLLM worker process
- `worker_multi_gpu.py`: llama.cpp multi-GPU launcher
- `worker_multi_gpu_vllm.py`: vLLM multi-GPU launcher
- `modal_worker.py`: local Modal relay worker
- `templates/`, `static/`: UI
- `tests/`: focused unit tests

## Local setup

```bash
uv sync
```

Run the backend:

```bash
uv run python app.py
```

Run the UI:

```bash
uv run python ui_app.py
```

## Worker modes

The router accepts two worker transport styles:

- `direct-endpoint`: the router calls the worker’s OpenAI-compatible endpoint directly
- `pull-request`: the worker polls the router for work and returns responses back through the worker API

The pull-request mode is the default for the bundled worker scripts.

## Worker images

Available Dockerfiles:

- `Dockerfile.worker-cuda`
- `Dockerfile.worker-cuda-multigpu`
- `Dockerfile.worker-vllm`
- `Dockerfile.worker-vllm-multigpu`

Build commands:

```bash
docker build -f Dockerfile.worker-cuda -t ghcr.io/seanmemery/llama-router-worker:latest .
docker build -f Dockerfile.worker-cuda-multigpu --build-arg BASE_IMAGE=ghcr.io/seanmemery/llama-router-worker:latest -t ghcr.io/seanmemery/llama-router-worker:multigpu .
docker build -f Dockerfile.worker-vllm -t ghcr.io/seanmemery/vllm-router-worker:latest .
docker build -f Dockerfile.worker-vllm-multigpu --build-arg BASE_IMAGE=ghcr.io/seanmemery/vllm-router-worker:latest -t ghcr.io/seanmemery/vllm-router-worker:multigpu .
```

## Common worker env

- `LLM_ROUTER_URL`
- `LLM_WORKER_TRANSPORT_MODE`
- `LLM_WORKER_PUBLIC_BASE_URL`
- `LLAMA_MODEL_PATH` or `LLAMA_MODEL_DIR`
- `LLAMA_MODEL_FILE`
- `LLAMA_MODEL_NAME`
- `LLAMA_PORT`
- `LLAMA_MAX_CONCURRENCY`

For vLLM workers launched from the repo checkout:

- `scripts/launch_vllm_worker.sh`

That script assumes the repo and model files are already present in the container or pod.

## Modal

If you want UI-launched Modal workers, create `.env` from `.env.example` and add:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Then run:

```bash
modal setup
```

## Tests

```bash
PYTHONPATH=. uv run --python .venv/bin/python pytest -q
```
