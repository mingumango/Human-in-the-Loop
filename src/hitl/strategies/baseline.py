"""Plain generation baseline strategy.

This is intentionally dataset-adapter driven. It supports GPQA multiple-choice
baseline runs and can also be used for simple GSM8K sanity checks.
"""

from __future__ import annotations

import time

from tqdm import tqdm

from hitl.config import ExperimentConfig
from hitl.datasets import get_dataset_adapter
from hitl.io import append_jsonl, existing_indices


def _validate_config(config: ExperimentConfig) -> None:
    if config.input_path is None:
        raise ValueError("--input is required.")
    if config.output_path is None and not config.dry_run:
        raise ValueError("--output is required unless --dry-run is set.")


def run(config: ExperimentConfig) -> None:
    _validate_config(config)

    adapter = get_dataset_adapter(config.dataset)
    examples = adapter.load(config.input_path, limit=config.limit)

    print(f"Loaded {len(examples)} item(s) from {config.input_path}")
    print(f"Dataset adapter: {adapter.name}")
    if config.dry_run:
        if examples:
            sample = examples[0]
            print("Dry run sample:")
            print(f"  index: {sample.index}")
            print(f"  question: {sample.question[:200]}")
            if sample.options:
                print(f"  options: {sample.options}")
            print(f"  ground_truth_final: {sample.ground_truth_final}")
        print("Dry run complete. Models were not loaded.")
        return

    from hitl.generation import generate_until_answer
    from hitl.models import load_reasoning_model

    print(f"Loading reasoning model: {config.model}")
    model, tokenizer = load_reasoning_model(config)
    already_done = existing_indices(config.output_path) if config.resume else set()

    for example in tqdm(examples, desc="Baseline"):
        if config.resume and example.index in already_done:
            continue

        prompt = adapter.build_prompt(example)
        start_time = time.time()
        generated_text = generate_until_answer(
            prompt,
            model,
            tokenizer,
            max_new_tokens=config.generation_max_new_tokens,
            temperature=config.generation_temperature,
        )
        duration_seconds = time.time() - start_time

        model_answer = adapter.extract_answer(generated_text)
        out_item = {
            **adapter.output_fields(example),
            "model_answer": model_answer,
            "correct": adapter.is_correct(model_answer, example),
            "generated_steps_raw": generated_text,
            "inference_time_sec": round(duration_seconds, 3),
        }
        append_jsonl(config.output_path, out_item)

    print(f"Done. Results written to: {config.output_path}")
