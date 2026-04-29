from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from config_loading import load_yaml_config
from paths import get_artifact_layout


class SharedLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str
    timeout: int = 0

    @model_validator(mode="after")
    def validate_fields(self) -> "SharedLLMConfig":
        if not self.model.strip():
            raise ValueError("LLM model must be non-empty")
        if not self.base_url.strip():
            raise ValueError("LLM base_url must be non-empty")
        if self.timeout < 0:
            raise ValueError("LLM timeout must be >= 0")
        return self


def default_llm_config_path() -> Path:
    return get_artifact_layout().config_root / "LLM.yaml"


def load_shared_llm_config(path: Path | None = None) -> SharedLLMConfig:
    data = load_yaml_config(path or default_llm_config_path())
    return SharedLLMConfig.model_validate(data)


def apply_shared_llm_defaults(
    data: dict[str, Any],
    *,
    model_key: str,
    base_url_key: str,
    timeout_key: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    if model_key in data and base_url_key in data and (timeout_key is None or timeout_key in data):
        return data
    llm = load_shared_llm_config(path)
    merged = dict(data)
    merged.setdefault(model_key, llm.model)
    merged.setdefault(base_url_key, llm.base_url)
    if timeout_key is not None:
        merged.setdefault(timeout_key, llm.timeout)
    return merged
