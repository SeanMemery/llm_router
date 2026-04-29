import modal

app = modal.App("qwen36-probe-direct")

image = (
    modal.Image.from_registry("vllm/vllm-openai:nightly", add_python="3.11")
    .entrypoint([])
    .run_commands(
        "uv pip install --system --upgrade huggingface-hub transformers==5.5.4 vllm==0.19.1",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    timeout=1800,
)
def run_probe() -> str:
    import vllm

    llm = vllm.LLM(
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        max_model_len=8192,
        attention_backend="flashinfer",
        async_scheduling=True,
    )
    sampling_params = llm.get_default_sampling_params()
    sampling_params.temperature = 0.0
    sampling_params.max_tokens = 16
    outputs = llm.chat(
        [[{"role": "user", "content": "Reply with exactly: modal real test ok"}]],
        sampling_params=sampling_params,
    )
    return str(outputs[0].outputs[0].text or "")


@app.local_entrypoint()
def main() -> None:
    result = run_probe.remote()
    print("RESULT:", result)
