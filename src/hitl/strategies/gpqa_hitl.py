"""GPQA HITL strategy with API trajectory reward.

This migrates ``archive/legacy_modules/gpqa_diamond/run_hitl_gpqa.py`` into the
new runner. Unlike GSM8K strategies that score each step with a PRM, this
strategy asks an OpenAI model to score the whole trajectory so far.
"""

from __future__ import annotations

import os
import time
from typing import Any

from tqdm import tqdm

from hitl.config import ExperimentConfig
from hitl.datasets import Example, get_dataset_adapter
from hitl.io import append_jsonl, existing_indices
from hitl.trace import TraceLogger


def generate_trajectory_reward_api(
    generated_steps: list[str],
    example: Example,
    reward_llm,
    logger: TraceLogger,
    openai_usage_acc: dict[str, float],
) -> float:
    if not generated_steps:
        return 0.0

    from langchain_community.callbacks import get_openai_callback

    from hitl.prompts import build_gpqa_trajectory_reward_prompt
    from hitl.schemas import TrajectoryRewardOutput

    prompt = build_gpqa_trajectory_reward_prompt(
        query=example.question,
        options=example.options or {},
        generated_steps=generated_steps,
    )

    result = None
    with get_openai_callback() as callback:
        try:
            result = reward_llm.with_structured_output(TrajectoryRewardOutput).invoke(prompt)
            score = float(result.score)
        except Exception as exc:
            logger.log(f"Error in reward generation: {exc}")
            score = 0.0

        openai_usage_acc["total_tokens"] += callback.total_tokens
        openai_usage_acc["total_cost"] += getattr(callback, "total_cost", 0.0) or 0.0

    logger.log("Generated trajectory reward")
    if result is not None:
        logger.log(f"Reward reasoning: {result.reasoning}")
    else:
        logger.log("Reward reasoning: reward model failed, default score 0.0")
    logger.log(f"Score: {score:.3f}")
    return score


def llm_intervention(
    example: Example,
    step_summary: str,
    feedback_llm,
    logger: TraceLogger,
    openai_usage_acc: dict[str, float],
):
    from langchain_community.callbacks import get_openai_callback

    from hitl.prompts import build_gpqa_feedback_prompt
    from hitl.schemas import GPQAFeedbackOutput, gpqa_feedback_schema

    prompt = build_gpqa_feedback_prompt(
        query=example.question,
        options=example.options or {},
        step_summary=step_summary,
        schema=gpqa_feedback_schema(),
    )

    with get_openai_callback() as callback:
        response = feedback_llm.with_structured_output(GPQAFeedbackOutput).invoke(prompt)
        openai_usage_acc["total_tokens"] += callback.total_tokens
        openai_usage_acc["total_cost"] += getattr(callback, "total_cost", 0.0) or 0.0

    logger.log(f"need_correction: {response.need_correction}")
    logger.log(f"unclear_step: {response.unclear_step}")
    logger.log(f"corrected_step: {response.corrected_step}")
    logger.log(f"reason: {response.reason}")
    return response


def llm_correction(
    example: Example,
    generated_steps: list[str],
    feedback,
    model,
    tokenizer,
    logger: TraceLogger,
    config: ExperimentConfig,
) -> str:
    from hitl.generation import generate_step
    from hitl.prompts import build_gpqa_correction_prompt

    correction_prompt = build_gpqa_correction_prompt(
        query=example.question,
        options=example.options or {},
        generated_steps=generated_steps,
        corrected_step=feedback.corrected_step,
    )
    corrected_step = generate_step(
        correction_prompt,
        model,
        tokenizer,
        max_new_tokens=config.correction_max_new_tokens,
        temperature=config.correction_temperature,
    )
    logger.log("Corrected reasoning step")
    logger.log(corrected_step)
    return corrected_step


def run_single_question(
    example: Example,
    model,
    tokenizer,
    reward_llm,
    feedback_llm,
    config: ExperimentConfig,
    logger: TraceLogger | None = None,
) -> dict[str, Any]:
    from hitl.generation import generate_step
    from hitl.prompts import build_gpqa_step_summary

    if logger is None:
        logger = TraceLogger()

    adapter = get_dataset_adapter("gpqa")
    full_prompt = adapter.build_prompt(example)
    generated_steps: list[str] = []
    generated_rewards: list[float] = []
    openai_usage_acc = {"total_tokens": 0, "total_cost": 0.0}
    time_tracking = {
        "generate_step_time": 0.0,
        "generate_reward_time": 0.0,
        "reflection_time": 0.0,
        "intervention_time": 0.0,
        "correction_time": 0.0,
    }

    logger.log("--- Starting GPQA HITL generation ---")

    for _ in range(config.max_steps):
        start_time = time.time()
        new_step = generate_step(
            full_prompt,
            model,
            tokenizer,
            max_new_tokens=config.generation_max_new_tokens,
            temperature=config.generation_temperature,
        )
        time_tracking["generate_step_time"] += time.time() - start_time
        logger.log(f"Generated Step {len(generated_steps) + 1}: {new_step}")

        generated_steps.append(new_step)

        start_time = time.time()
        current_score = generate_trajectory_reward_api(
            generated_steps,
            example,
            reward_llm,
            logger,
            openai_usage_acc,
        )
        time_tracking["generate_reward_time"] += time.time() - start_time
        logger.log(f"Current trajectory score: {current_score:.3f}")

        correction_attempts = 0
        while (
            current_score <= config.reward_threshold
            and correction_attempts < config.max_corrections
        ):
            correction_attempts += 1
            logger.log("Low score detected. Triggering GPQA tutor intervention.")

            step_summary = build_gpqa_step_summary(
                generated_steps,
                generated_rewards,
                current_score,
            )

            start_time = time.time()
            feedback = llm_intervention(
                example,
                step_summary,
                feedback_llm,
                logger,
                openai_usage_acc,
            )
            time_tracking["intervention_time"] += time.time() - start_time

            if feedback.need_correction:
                unclear_step = feedback.unclear_step
                if 1 <= unclear_step <= len(generated_steps):
                    cut_idx = unclear_step
                else:
                    cut_idx = len(generated_steps)

                generated_steps = generated_steps[:cut_idx]
                generated_rewards = generated_rewards[: cut_idx - 1] if cut_idx > 1 else []

            if not feedback.need_correction:
                current_score = 0.9
                break

            start_time = time.time()
            corrected = llm_correction(
                example,
                generated_steps,
                feedback,
                model,
                tokenizer,
                logger,
                config,
            )
            time_tracking["correction_time"] += time.time() - start_time

            if generated_steps:
                generated_steps[-1] = corrected
            else:
                generated_steps.append(corrected)

            new_step = corrected

            start_time = time.time()
            current_score = generate_trajectory_reward_api(
                generated_steps,
                example,
                reward_llm,
                logger,
                openai_usage_acc,
            )
            time_tracking["generate_reward_time"] += time.time() - start_time
            logger.log(f"Corrected trajectory score: {current_score:.3f}")

        generated_rewards.append(float(current_score))
        full_prompt += new_step + "\n"

        if "</answer>" in new_step.lower():
            logger.log("--- Generation stopped: answer found ---")
            break

    model_answer = adapter.extract_answer("\n".join(generated_steps))
    result = {
        "generated_steps": generated_steps,
        "generated_rewards": generated_rewards,
        "model_answer": model_answer,
        "trace": logger.dump(),
        "openai_total_tokens": int(openai_usage_acc["total_tokens"]),
        "openai_total_cost_usd": float(openai_usage_acc["total_cost"]),
    }
    result.update({key: round(value, 3) for key, value in time_tracking.items()})
    return result


def _validate_config(config: ExperimentConfig) -> None:
    if (config.dataset or "").lower() not in {"gpqa", "gpqa_diamond"}:
        raise ValueError("gpqa_hitl requires --dataset gpqa.")
    if config.input_path is None:
        raise ValueError("--input is required.")
    if config.output_path is None and not config.dry_run:
        raise ValueError("--output is required unless --dry-run is set.")


def _build_openai_llm(model_name: str, temperature: float, max_tokens: int):
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        max_tokens=max_tokens,
    )


def run(config: ExperimentConfig) -> None:
    _validate_config(config)

    adapter = get_dataset_adapter(config.dataset)
    examples = adapter.load(config.input_path, limit=config.limit)

    print(f"Loaded {len(examples)} item(s) from {config.input_path}")
    print("Dataset adapter: gpqa")
    print(f"Trajectory reward model: {config.reward_model}")
    print(f"Feedback model: {config.openai_model}")
    if config.dry_run:
        if examples:
            sample = examples[0]
            print("Dry run sample:")
            print(f"  index: {sample.index}")
            print(f"  question: {sample.question[:200]}")
            print(f"  options: {sample.options}")
            print(f"  ground_truth_final: {sample.ground_truth_final}")
        print("Dry run complete. Models and OpenAI clients were not loaded.")
        return

    from hitl.models import load_reasoning_model

    print(f"Loading reasoning model: {config.model}")
    model, tokenizer = load_reasoning_model(config)
    reward_llm = _build_openai_llm(
        model_name=config.reward_model or "gpt-5-mini",
        temperature=0.0,
        max_tokens=3000,
    )
    feedback_llm = _build_openai_llm(
        model_name=config.openai_model,
        temperature=config.openai_temperature,
        max_tokens=config.openai_max_tokens,
    )

    already_done = existing_indices(config.output_path) if config.resume else set()

    for example in tqdm(examples, desc="GPQA HITL"):
        if config.resume and example.index in already_done:
            continue

        start_time = time.time()
        result = run_single_question(
            example=example,
            model=model,
            tokenizer=tokenizer,
            reward_llm=reward_llm,
            feedback_llm=feedback_llm,
            config=config,
            logger=TraceLogger(),
        )
        duration_seconds = time.time() - start_time
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)

        out_item = {
            **adapter.output_fields(example),
            "generated_steps": result["generated_steps"],
            "generated_rewards": result["generated_rewards"],
            "model_answer": result["model_answer"],
            "correct": adapter.is_correct(result["model_answer"], example),
            "trace": result["trace"],
            "openai_total_tokens": result["openai_total_tokens"],
            "openai_total_cost_usd": result["openai_total_cost_usd"],
            "processing_time_seconds": round(duration_seconds, 2),
            "processing_time_str": f"{minutes}m {seconds}s",
            "generate_step_time": result.get("generate_step_time", 0.0),
            "generate_reward_time": result.get("generate_reward_time", 0.0),
            "reflection_time": result.get("reflection_time", 0.0),
            "intervention_time": result.get("intervention_time", 0.0),
            "correction_time": result.get("correction_time", 0.0),
        }
        append_jsonl(config.output_path, out_item)

    print(f"Done. Results written to: {config.output_path}")
