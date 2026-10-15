from __future__ import annotations

from pathlib import Path
from typing import Any


def _remove_absolute_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_absolute_paths(item)
            for key, item in value.items()
            if key not in {"tokenizer_file", "name_or_path", "_name_or_path"}
        }
    if isinstance(value, list):
        return [_remove_absolute_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_remove_absolute_paths(item) for item in value)
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def sanitize_huggingface_metadata(
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> None:
    if model is not None:
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is None:
                continue
            if hasattr(config, "_name_or_path"):
                config._name_or_path = ""
            if hasattr(config, "name_or_path"):
                config.name_or_path = ""

    if tokenizer is not None:
        if hasattr(tokenizer, "name_or_path"):
            tokenizer.name_or_path = ""
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            tokenizer.init_kwargs = _remove_absolute_paths(init_kwargs)
