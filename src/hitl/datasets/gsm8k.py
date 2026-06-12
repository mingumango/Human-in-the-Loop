"""GSM8K-style JSONL adapter."""

from __future__ import annotations

import re
from pathlib import Path

from hitl.evaluation import numeric_answer_correct
from hitl.io import read_jsonl
from hitl.prompts import build_initial_prompt

from .base import Example


class GSM8KAdapter:
    name = "gsm8k"

    def load(self, path: str | Path, limit: int | None = None) -> list[Example]:
        examples = []
        for row_idx, row in enumerate(read_jsonl(path)):
            examples.append(
                Example(
                    index=row.get("index", row_idx),
                    question=row.get("question", ""),
                    ground_truth_text=row.get("ground_truth_text", ""),
                    ground_truth_final=row.get("ground_truth_final"),
                    raw=row,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
        return examples

    def build_prompt(self, example: Example) -> str:
        return build_initial_prompt(example.question)

    def extract_answer(self, generated_text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def is_correct(self, model_answer: str, example: Example) -> bool:
        return numeric_answer_correct(model_answer, example.ground_truth_final)

    def output_fields(self, example: Example) -> dict:
        return {
            "index": example.index,
            "question": example.question,
            "ground_truth_text": example.ground_truth_text,
            "ground_truth_final": example.ground_truth_final,
        }
