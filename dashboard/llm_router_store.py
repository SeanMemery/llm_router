from __future__ import annotations

import json
from pathlib import Path

from dashboard.llm_router import LLMRouterSnapshot
from paths import ArtifactLayout


def llm_router_store_path(layout: ArtifactLayout) -> Path:
    return layout.data_root / "dashboard" / "llm_router_state.json"


def load_llm_router_snapshot(layout: ArtifactLayout) -> LLMRouterSnapshot:
    path = llm_router_store_path(layout)
    if not path.is_file():
        return LLMRouterSnapshot()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed dashboard LLM router store: {path}")
    return LLMRouterSnapshot.model_validate(payload)


def save_llm_router_snapshot(layout: ArtifactLayout, snapshot: LLMRouterSnapshot) -> Path:
    path = llm_router_store_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
