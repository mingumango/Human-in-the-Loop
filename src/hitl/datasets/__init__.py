"""Dataset adapters for experiment runners."""

from __future__ import annotations

from .base import DatasetAdapter, Example
from .gpqa import GPQAAdapter
from .gsm8k import GSM8KAdapter


def get_dataset_adapter(name: str | None) -> DatasetAdapter:
    normalized = (name or "gsm8k").lower()
    if normalized in {"gsm8k", "math"}:
        return GSM8KAdapter()
    if normalized in {"gpqa", "gpqa_diamond"}:
        return GPQAAdapter()
    raise ValueError(f"Unsupported dataset adapter: {name}")


__all__ = ["DatasetAdapter", "Example", "get_dataset_adapter"]
