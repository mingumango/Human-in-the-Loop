"""GPQA multiple-choice adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hitl.evaluation import choice_answer_correct, extract_choice_letter
from hitl.io import read_jsonl

from .base import Example


class GPQAAdapter:
    name = "gpqa"

    def load(self, path: str | Path, limit: int | None = None) -> list[Example]:
        input_path = Path(path)
        if input_path.is_dir():
            rows = self._load_disk_rows(input_path)
        else:
            rows = list(read_jsonl(input_path))

        examples = []
        for row_idx, row in enumerate(rows):
            examples.append(self._example_from_row(row, row_idx))
            if limit is not None and len(examples) >= limit:
                break
        return examples

    def _load_disk_rows(self, path: Path) -> list[dict[str, Any]]:
        try:
            from datasets import load_from_disk

            return [dict(row) for row in load_from_disk(str(path))]
        except ImportError:
            return self._load_arrow_rows(path)

    def _load_arrow_rows(self, path: Path) -> list[dict[str, Any]]:
        try:
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise RuntimeError(
                "Loading GPQA disk datasets requires either datasets or pyarrow."
            ) from exc

        state_path = path / "state.json"
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            data_files = [
                path / data_file["filename"]
                for data_file in state.get("_data_files", [])
                if "filename" in data_file
            ]
        else:
            data_files = sorted(path.glob("*.arrow"))

        rows: list[dict[str, Any]] = []
        for data_file in data_files:
            with ipc.open_stream(str(data_file)) as reader:
                rows.extend(reader.read_all().to_pylist())
        return rows

    def _example_from_row(self, row: dict[str, Any], row_idx: int) -> Example:
        options = row.get("options")
        if not isinstance(options, dict):
            options = {
                "A": row.get("A", ""),
                "B": row.get("B", ""),
                "C": row.get("C", ""),
                "D": row.get("D", ""),
            }

        gold = (
            row.get("answer")
            or row.get("ground_truth")
            or row.get("ground_truth_final")
            or row.get("gold")
        )

        return Example(
            index=row.get("index", row_idx),
            question=row.get("question", ""),
            ground_truth_final=str(gold).strip().upper() if gold is not None else None,
            ground_truth_text=str(gold).strip().upper() if gold is not None else "",
            options={key: str(value) for key, value in options.items()},
            raw=row,
        )

    def build_prompt(self, example: Example) -> str:
        options = example.options or {}
        return f"""
You are a helpful science tutor. You will be given a multiple-choice question.
Think step by step and show your reasoning inside <step>...</step> tags.
At the end, choose exactly one option among A, B, C, and D.

The last line of your answer MUST be in the format:
<answer> LETTER </answer>
where LETTER is one of A, B, C, or D.

Question:
{example.question}

Options:
A) {options.get("A", "")}
B) {options.get("B", "")}
C) {options.get("C", "")}
D) {options.get("D", "")}

Solution:
""".lstrip()

    def extract_answer(self, generated_text: str) -> str:
        return extract_choice_letter(generated_text) or ""

    def is_correct(self, model_answer: str, example: Example) -> bool:
        return choice_answer_correct(model_answer, example.ground_truth_final)

    def output_fields(self, example: Example) -> dict[str, Any]:
        return {
            "index": example.index,
            "question": example.question,
            "options": example.options or {},
            "ground_truth_final": example.ground_truth_final,
        }
