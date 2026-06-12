"""Configuration objects for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str | None = "gsm8k"
    strategy: str | None = "no_reflection"
    input_path: Path | None = None
    output_path: Path | None = None
    model: str | None = "meta-llama/Llama-3.1-8B-Instruct"
    reward_model: str | None = "Qwen/Qwen2.5-Math-PRM-7B"
    max_steps: int = 10
    num_samples: int = 5
    resume: bool = False
    limit: int | None = None
    dry_run: bool = False
    reward_threshold: float = 0.5
    max_corrections: int = 3
    generation_max_new_tokens: int = 150
    generation_temperature: float = 0.5
    correction_max_new_tokens: int = 200
    correction_temperature: float = 0.7
    reflection_max_new_tokens: int = 400
    reflection_temperature: float = 0.5
    global_feedback_mode: str = "hf"
    openai_model: str = "gpt-5-mini"
    openai_temperature: float = 0.5
    openai_max_tokens: int = 5000
    reasoning_dtype: str = "bfloat16"
    reasoning_load_in_8bit: bool = False
    reward_load_in_4bit: bool = True
