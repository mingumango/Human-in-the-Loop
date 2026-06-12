"""JSONL helpers shared by experiment runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON objects from a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file, creating parents if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def existing_indices(path: str | Path, key: str = "index") -> set[Any]:
    """Return index values already present in an output JSONL file."""
    output_path = Path(path)
    if not output_path.exists():
        return set()

    done: set[Any] = set()
    for obj in read_jsonl(output_path):
        if key in obj:
            done.add(obj[key])
    return done
