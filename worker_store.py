from __future__ import annotations

import json
from pathlib import Path

from paths import ArtifactLayout
from worker_registry import LLMWorkerRecord


def llm_worker_store_path(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "llm_router_workers.json"


def load_llm_workers(layout: ArtifactLayout) -> list[LLMWorkerRecord]:
    path = llm_worker_store_path(layout)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Malformed worker store: {path}")
    return [LLMWorkerRecord.model_validate(item) for item in payload if isinstance(item, dict)]


def save_llm_workers(layout: ArtifactLayout, workers: list[LLMWorkerRecord]) -> Path:
    path = llm_worker_store_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([item.model_dump(mode="json") for item in workers], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
