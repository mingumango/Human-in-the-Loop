"""Trace capture used by experiment strategies."""

from __future__ import annotations


class TraceLogger:
    """Small in-memory logger that preserves per-item traces in outputs."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)

    def extend(self, lines: list[str]) -> None:
        self.messages.extend(lines)

    def dump(self) -> list[str]:
        return list(self.messages)
