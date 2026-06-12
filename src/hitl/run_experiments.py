"""Unified entry point for HITL experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a HITL experiment.")
    parser.add_argument("--config", type=Path, help="Path to a YAML/JSON config file.")
    parser.add_argument("--dataset", help="Dataset adapter, e.g. gsm8k or gpqa.")
    parser.add_argument("--strategy", help="Experiment strategy, e.g. no_reflection.")
    parser.add_argument("--input", dest="input_path", type=Path, help="Input JSONL path.")
    parser.add_argument("--output", dest="output_path", type=Path, help="Output JSONL path.")
    parser.add_argument("--model", help="Reasoning model name or local path.")
    parser.add_argument("--reward-model", dest="reward_model", help="Reward model name or path.")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-samples", type=int, help="Number of paths for self_consistency.")
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--limit", type=int, help="Process at most N input rows.")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Validate config/data without loading models.")
    parser.add_argument("--reward-threshold", type=float, help="Trigger feedback when latest PRM score is <= this value.")
    parser.add_argument("--max-corrections", type=int, help="Maximum correction attempts per generated step.")
    parser.add_argument("--openai-model", help="Feedback LLM name.")
    parser.add_argument("--openai-temperature", type=float)
    parser.add_argument("--openai-max-tokens", type=int)
    parser.add_argument("--generation-max-new-tokens", type=int)
    parser.add_argument("--generation-temperature", type=float)
    parser.add_argument("--correction-max-new-tokens", type=int)
    parser.add_argument("--correction-temperature", type=float)
    parser.add_argument("--reflection-max-new-tokens", type=int)
    parser.add_argument("--reflection-temperature", type=float)
    parser.add_argument(
        "--global-feedback-mode",
        choices=["hf", "hf_no_reason", "hc", "sf"],
        help="Global feedback variant: hf, hf_no_reason, hc, or sf.",
    )
    parser.add_argument("--reasoning-dtype", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--8bit",
        dest="reasoning_load_in_8bit",
        action="store_true",
        default=None,
        help="Load the main reasoning model in 8-bit.",
    )
    parser.add_argument(
        "--no-reward-4bit",
        dest="reward_load_in_4bit",
        action="store_false",
        default=None,
        help="Load reward model without 4-bit quantization.",
    )
    return parser.parse_args()


def _load_config_file(path: Path) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML config files.") from exc
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config file type: {path.suffix}")


def _normalize_config_keys(raw: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "input": "input_path",
        "output": "output_path",
        "reward-model": "reward_model",
        "max-steps": "max_steps",
        "dry-run": "dry_run",
        "reward-threshold": "reward_threshold",
        "max-corrections": "max_corrections",
        "openai-model": "openai_model",
        "openai-temperature": "openai_temperature",
        "openai-max-tokens": "openai_max_tokens",
        "reflection-max-new-tokens": "reflection_max_new_tokens",
        "reflection-temperature": "reflection_temperature",
        "global-feedback-mode": "global_feedback_mode",
    }
    normalized = {}
    field_names = {field.name for field in fields(ExperimentConfig)}
    for key, value in raw.items():
        normalized_key = aliases.get(key, key)
        if normalized_key not in field_names:
            raise ValueError(f"Unknown config key: {key}")
        normalized[normalized_key] = value

    for path_key in ("input_path", "output_path"):
        if normalized.get(path_key) is not None:
            normalized[path_key] = Path(normalized[path_key])

    return normalized


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "dataset",
        "strategy",
        "input_path",
        "output_path",
        "model",
        "reward_model",
        "max_steps",
        "num_samples",
        "resume",
        "limit",
        "dry_run",
        "reward_threshold",
        "max_corrections",
        "generation_max_new_tokens",
        "generation_temperature",
        "correction_max_new_tokens",
        "correction_temperature",
        "reflection_max_new_tokens",
        "reflection_temperature",
        "global_feedback_mode",
        "openai_model",
        "openai_temperature",
        "openai_max_tokens",
        "reasoning_dtype",
        "reasoning_load_in_8bit",
        "reward_load_in_4bit",
    ]
    return {key: getattr(args, key) for key in keys if getattr(args, key) is not None}


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    raw: dict[str, Any] = {}
    if args.config is not None:
        raw.update(_normalize_config_keys(_load_config_file(args.config)))
    raw.update(_cli_overrides(args))
    return ExperimentConfig(**raw)


def dispatch(config: ExperimentConfig) -> None:
    strategy = (config.strategy or "").lower()
    if strategy == "baseline":
        from .strategies.baseline import run

        run(config)
        return

    if strategy == "gpqa_hitl":
        from .strategies.gpqa_hitl import run

        run(config)
        return

    if strategy in {"no_reflection", "wo_reflection"}:
        from .strategies.no_reflection import run

        run(config)
        return

    if strategy == "reflection":
        from .strategies.reflection import run

        run(config)
        return

    if strategy == "global_feedback":
        from .strategies.global_feedback import run

        run(config)
        return

    if strategy == "self_consistency":
        from .strategies.self_consistency import run

        run(config)
        return

    raise ValueError(
        f"Unsupported strategy: {config.strategy}. "
        "Currently migrated strategies: baseline, gpqa_hitl, no_reflection, reflection, "
        "global_feedback, self_consistency."
    )


def main() -> None:
    config = build_config(parse_args())
    print("Parsed experiment config:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    dispatch(config)


if __name__ == "__main__":
    main()
