from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    return _load_yaml_config(path.resolve(), stack=[])


def _load_yaml_config(path: Path, *, stack: list[Path]) -> dict[str, Any]:
    if path in stack:
        cycle = " -> ".join(str(item) for item in [*stack, path])
        raise ValueError(f"Config extends cycle detected: {cycle}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML object: {path}")

    extends = data.pop("extends", None)
    if extends is None:
        return data
    if not isinstance(extends, str) or not extends.strip():
        raise ValueError(f"Config 'extends' must be a non-empty string: {path}")

    base_path = (path.parent / extends).resolve()
    base_data = _load_yaml_config(base_path, stack=[*stack, path])
    return _deep_merge(base_data, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
