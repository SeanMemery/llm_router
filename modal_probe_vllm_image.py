import modal

app = modal.App("vllm-image-probe")

image = (
    modal.Image.from_registry("vllm/vllm-openai:nightly", add_python="3.11")
    .entrypoint([])
    .run_commands(
        "uv pip install --system --upgrade huggingface-hub transformers==5.5.4 vllm==0.19.1",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(image=image, gpu="H100", timeout=900)
def run_probe() -> str:
    import vllm

    return str(getattr(vllm, "__version__", "unknown"))


@app.local_entrypoint()
def main() -> None:
    result = run_probe.remote()
    print("RESULT:", result)
