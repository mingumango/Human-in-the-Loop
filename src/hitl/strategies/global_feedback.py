"""Global-feedback HITL strategy.

This consolidates the legacy ``run_cot_hitl_global_feedback_*`` scripts.

Modes:
- ``hf``: tutor gives a corrected step, reason, and global advice; the model
  generates a correction from that feedback.
- ``hf_no_reason``: like ``hf``, but the correction prompt omits the reason.
- ``hc``: tutor gives a corrected step and the strategy applies it directly.
- ``sf``: tutor gives local feedback and global advice; the model generates a
  correction from the feedback.
"""

from __future__ import annotations

import time
from typing import Any

from tqdm import tqdm

from hitl.config import ExperimentConfig
from hitl.evaluation import numeric_answer_correct
from hitl.io import append_jsonl, existing_indices, read_jsonl
from hitl.trace import TraceLogger


DIRECT_MODES = {"hf", "hf_no_reason", "hc"}
SOFT_MODES = {"sf"}


def _latest_score(step_reward: list[list[float]]) -> float:
    if not step_reward or not step_reward[0]:
        return 0.0
    return float(step_reward[0][-1])


def _mode(config: ExperimentConfig) -> str:
    mode = config.global_feedback_mode.lower()
    if mode not in DIRECT_MODES | SOFT_MODES:
        raise ValueError(
            f"Unsupported global_feedback_mode: {config.global_feedback_mode}. "
            "Use one of: hf, hf_no_reason, hc, sf."
        )
    return mode


def llm_intervention(
    query: str,
    step_summary: str,
    reflection: str,
    feedback_llm,
    logger: TraceLogger,
    openai_usage_acc: dict[str, float],
    mode: str,
):
    from langchain_community.callbacks import get_openai_callback

    from hitl.prompts import build_global_direct_feedback_prompt
    from hitl.prompts import build_global_soft_feedback_prompt
    from hitl.schemas import GlobalDirectFeedbackOutput, GlobalSoftFeedbackOutput
    from hitl.schemas import global_direct_feedback_schema, global_soft_feedback_schema

    if mode in DIRECT_MODES:
        prompt = build_global_direct_feedback_prompt(
            query=query,
            step_summary=step_summary,
            reflection=reflection,
            schema=global_direct_feedback_schema(),
        )
        output_type = GlobalDirectFeedbackOutput
    else:
        prompt = build_global_soft_feedback_prompt(
            query=query,
            step_summary=step_summary,
            reflection=reflection,
            schema=global_soft_feedback_schema(),
        )
        output_type = GlobalSoftFeedbackOutput

    with get_openai_callback() as callback:
        response = feedback_llm.with_structured_output(output_type).invoke(prompt)
        openai_usage_acc["total_tokens"] += callback.total_tokens
        openai_usage_acc["total_cost"] += getattr(callback, "total_cost", 0.0) or 0.0

    logger.log(f"need_correction: {response.need_correction}")
    logger.log(f"unclear_step: {response.unclear_step}")
    if mode in DIRECT_MODES:
        logger.log(f"corrected_step: {response.corrected_step}")
        logger.log(f"reason: {response.reason}")
    else:
        logger.log(f"local_feedback: {response.feedback}")
    logger.log(f"global_advice: {response.global_advice}")
    return response


def _generate_direct_correction(
    query: str,
    generated_steps: list[str],
    feedback,
    model,
    tokenizer,
    logger: TraceLogger,
    config: ExperimentConfig,
    include_reason: bool,
) -> str:
    from hitl.generation import generate_step
    from hitl.prompts import build_global_direct_correction_prompt

    prompt = build_global_direct_correction_prompt(
        query=query,
        generated_steps=generated_steps,
        corrected_step=feedback.corrected_step,
        reason=feedback.reason if include_reason else "",
        global_advice=feedback.global_advice,
    )
    corrected_step = generate_step(
        prompt,
        model,
        tokenizer,
        max_new_tokens=config.correction_max_new_tokens,
        temperature=config.correction_temperature,
    )
    logger.log("Corrected reasoning step")
    logger.log(corrected_step)
    return corrected_step


def _generate_soft_correction(
    query: str,
    generated_steps: list[str],
    feedback,
    model,
    tokenizer,
    logger: TraceLogger,
    config: ExperimentConfig,
) -> str:
    from hitl.generation import generate_step
    from hitl.prompts import build_global_soft_correction_prompt

    prompt = build_global_soft_correction_prompt(
        query=query,
        generated_steps=generated_steps,
        feedback=feedback.feedback,
        global_advice=feedback.global_advice,
    )
    corrected_step = generate_step(
        prompt,
        model,
        tokenizer,
        max_new_tokens=config.correction_max_new_tokens,
        temperature=config.correction_temperature,
    )
    logger.log("Corrected reasoning step")
    logger.log(corrected_step)
    return corrected_step


def run_single_question(
    query: str,
    model,
    tokenizer,
    reward_model,
    reward_tokenizer,
    feedback_llm,
    config: ExperimentConfig,
    logger: TraceLogger | None = None,
) -> dict[str, Any]:
    from hitl.generation import extract_final_answer_text, generate_step
    from hitl.prompts import build_prompt_with_context, build_step_summary
    from hitl.rewards import generate_reward
    from hitl.strategies.reflection import llm_reflection

    if logger is None:
        logger = TraceLogger()

    mode = _mode(config)
    latest_global_advice = ""
    generated_steps: list[str] = []
    last_step_scores: list[float] = []
    openai_usage_acc = {"total_tokens": 0, "total_cost": 0.0}
    time_tracking = {
        "generate_step_time": 0.0,
        "generate_reward_time": 0.0,
        "reflection_time": 0.0,
        "intervention_time": 0.0,
        "correction_time": 0.0,
    }

    logger.log("--- Starting multi-step generation ---")

    while len(generated_steps) < config.max_steps:
        base_prompt = build_prompt_with_context(query, latest_global_advice)
        if generated_steps:
            full_prompt = base_prompt + "\n" + "\n".join(generated_steps) + "\n"
        else:
            full_prompt = base_prompt

        start_time = time.time()
        new_step = generate_step(
            full_prompt,
            model,
            tokenizer,
            max_new_tokens=config.generation_max_new_tokens,
            temperature=config.generation_temperature,
        )
        time_tracking["generate_step_time"] += time.time() - start_time

        if not new_step.strip():
            logger.log("Empty step generated. Stopping.")
            break

        generated_steps.append(new_step)
        logger.log(f"Generated Step {len(generated_steps)}: {new_step}")

        start_time = time.time()
        step_reward = generate_reward(generated_steps, query, reward_model, reward_tokenizer)
        time_tracking["generate_reward_time"] += time.time() - start_time
        last_score = _latest_score(step_reward)
        logger.log(f"Step {len(generated_steps)} Reward: {last_score:.3f}")

        correction_attempts = 0
        is_final_answer = "</answer>" in new_step.lower()
        while (
            last_score <= config.reward_threshold
            and correction_attempts < config.max_corrections
            and not is_final_answer
        ):
            correction_attempts += 1
            logger.log(
                f"Low reward detected. Triggering global feedback attempt {correction_attempts}."
            )

            step_summary = build_step_summary(generated_steps, step_reward)

            start_time = time.time()
            reflection = llm_reflection(
                generated_steps,
                step_reward,
                query,
                model,
                tokenizer,
                logger,
                config,
            )
            time_tracking["reflection_time"] += time.time() - start_time

            start_time = time.time()
            feedback = llm_intervention(
                query,
                step_summary,
                reflection,
                feedback_llm,
                logger,
                openai_usage_acc,
                mode,
            )
            time_tracking["intervention_time"] += time.time() - start_time

            if feedback.global_advice:
                latest_global_advice = feedback.global_advice
                logger.log(f"Updated global advice: {latest_global_advice}")

            if not feedback.need_correction:
                last_score = 0.9
                break

            if mode == "hc":
                target_idx = feedback.unclear_step - 1
                if target_idx < 0 or target_idx >= len(generated_steps):
                    logger.log(f"Invalid step index {feedback.unclear_step}. Ignoring correction.")
                    break

                start_time = time.time()
                generated_steps = generated_steps[:target_idx]
                last_step_scores = last_step_scores[:target_idx]
                corrected = feedback.corrected_step.strip()
                generated_steps.append(corrected)
                new_step = corrected
                time_tracking["correction_time"] += time.time() - start_time
                logger.log(f"Applied tutor correction at Step {target_idx + 1}")
            else:
                unclear_step = feedback.unclear_step
                generated_steps = generated_steps[:unclear_step]
                last_step_scores = last_step_scores[: unclear_step - 1] if unclear_step > 1 else []

                start_time = time.time()
                if mode in {"hf", "hf_no_reason"}:
                    corrected = _generate_direct_correction(
                        query,
                        generated_steps,
                        feedback,
                        model,
                        tokenizer,
                        logger,
                        config,
                        include_reason=(mode == "hf"),
                    )
                else:
                    corrected = _generate_soft_correction(
                        query,
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
            step_reward = generate_reward(generated_steps, query, reward_model, reward_tokenizer)
            time_tracking["generate_reward_time"] += time.time() - start_time
            last_score = _latest_score(step_reward)
            logger.log(f"Corrected reward: {last_score:.3f}")

            if last_score > config.reward_threshold:
                logger.log("Reward improved. Accepting correction.")
                break

            is_final_answer = "</answer>" in new_step.lower()

        last_step_scores.append(float(last_score))

        if "</answer>" in new_step.lower():
            logger.log("--- Generation stopped: answer found ---")
            break

    result = {
        "generated_steps": generated_steps,
        "generated_rewards": last_step_scores,
        "model_answer": extract_final_answer_text(generated_steps) or "",
        "trace": logger.dump(),
        "openai_total_tokens": int(openai_usage_acc["total_tokens"]),
        "openai_total_cost_usd": float(openai_usage_acc["total_cost"]),
        "final_global_advice": latest_global_advice,
    }
    result.update({k: round(v, 3) for k, v in time_tracking.items()})
    return result


def _validate_config(config: ExperimentConfig) -> None:
    if config.input_path is None:
        raise ValueError("--input is required.")
    if config.output_path is None and not config.dry_run:
        raise ValueError("--output is required unless --dry-run is set.")
    _mode(config)


def run(config: ExperimentConfig) -> None:
    _validate_config(config)

    dataset = list(read_jsonl(config.input_path))
    if config.limit is not None:
        dataset = dataset[: config.limit]

    print(f"Loaded {len(dataset)} item(s) from {config.input_path}")
    print(f"Global feedback mode: {_mode(config)}")
    if config.dry_run:
        if dataset:
            sample = dataset[0]
            print("Dry run sample:")
            print(f"  index: {sample.get('index')}")
            print(f"  question: {sample.get('question', '')[:200]}")
        print("Dry run complete. Models were not loaded.")
        return

    from hitl.models import build_feedback_llm, load_reasoning_model, load_reward_model

    print(f"Loading reasoning model: {config.model}")
    model, tokenizer = load_reasoning_model(config)
    print(f"Loading reward model: {config.reward_model}")
    reward_model, reward_tokenizer = load_reward_model(config)
    feedback_llm = build_feedback_llm(config)

    already_done = existing_indices(config.output_path) if config.resume else set()

    for obj in tqdm(dataset, desc="Processing Questions"):
        idx = obj.get("index")
        if config.resume and idx in already_done:
            continue

        start_time = time.time()
        question = obj.get("question", "")
        gt_text = obj.get("ground_truth_text", "")
        gt_final = obj.get("ground_truth_final", None)

        res = run_single_question(
            query=question,
            model=model,
            tokenizer=tokenizer,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            feedback_llm=feedback_llm,
            config=config,
            logger=TraceLogger(),
        )

        duration_seconds = time.time() - start_time
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        model_answer_text = res.get("model_answer") or ""

        out_item = {
            "index": idx,
            "question": question,
            "ground_truth_text": gt_text,
            "ground_truth_final": gt_final,
            "generated_steps": res["generated_steps"],
            "generated_rewards": res["generated_rewards"],
            "model_answer": model_answer_text,
            "correct": numeric_answer_correct(model_answer_text, gt_final),
            "trace": res["trace"],
            "openai_total_tokens": res["openai_total_tokens"],
            "openai_total_cost_usd": res["openai_total_cost_usd"],
            "final_global_advice": res.get("final_global_advice", ""),
            "processing_time_seconds": round(duration_seconds, 2),
            "processing_time_str": f"{minutes}m {seconds}s",
            "generate_step_time": res.get("generate_step_time", 0.0),
            "generate_reward_time": res.get("generate_reward_time", 0.0),
            "reflection_time": res.get("reflection_time", 0.0),
            "intervention_time": res.get("intervention_time", 0.0),
            "correction_time": res.get("correction_time", 0.0),
        }
        append_jsonl(config.output_path, out_item)

    print(f"Done. Results written to: {config.output_path}")
