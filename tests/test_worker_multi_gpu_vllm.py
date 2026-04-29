from __future__ import annotations

from worker_multi_gpu_vllm import (
    _parse_gpu_indices,
    _public_base_url_for_gpu,
)


def test_parse_gpu_indices_discards_empty_values_vllm() -> None:
    assert _parse_gpu_indices("0, 1, ,2") == ["0", "1", "2"]


def test_public_base_url_template_expands_gpu_and_port_vllm() -> None:
    assert (
        _public_base_url_for_gpu(
            template="http://worker-{gpu}.svc:{port}",
            gpu_index="3",
            port=8091,
            worker_index=1,
        )
        == "http://worker-3.svc:8091"
    )
