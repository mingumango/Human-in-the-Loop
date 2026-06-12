"""Dataset adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Example:
    index: Any
    question: str
    ground_truth_final: Any = None
    ground_truth_text: str = ""
    options: dict[str, str] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    name: str

    def load(self, path: str | Path, limit: int | None = None) -> list[Example]:
        ...

    def build_prompt(self, example: Example) -> str:
        ...

    def extract_answer(self, generated_text: str) -> str:
        ...

    def is_correct(self, model_answer: str, example: Example) -> bool:
        ...

    def output_fields(self, example: Example) -> dict[str, Any]:
        ...
