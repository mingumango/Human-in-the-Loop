"""Reflection-based HITL strategy.

This migrates ``archive/legacy_scripts/run_cot_hitl_reflection_llama.py``:
when a generated step receives a low PRM score, the reasoning model first
produces a self-reflection, then a feedback LLM uses that reflection to provide
direct correction guidance.
"""

from __future__ import annotations

import re
import time
from typing import Any

from tqdm import tqdm

from hitl.config import ExperimentConfig
from hitl.evaluation import numeric_answer_correct
from hitl.io import append_jsonl, existing_indices, read_jsonl
from hitl.trace import TraceLogger


def _latest_score(step_reward: list[list[float]]) -> float:
    if not step_reward or not step_reward[0]:
        return 0.0
    return float(step_reward[0][-1])


def llm_reflection(
    generated_steps: list[str],
    step_reward: list[list[float]],
    query: str,
    model,
    tokenizer,
    logger: TraceLogger,
    config: ExperimentConfig,
) -> str:
    import torch

    from hitl.prompts import build_reflection_prompt, build_step_summary

    step_summary = build_step_summary(generated_steps, step_reward)
    reflection_prompt = build_reflection_prompt(query, step_summary)
    inputs = tokenizer(reflection_prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    generation_kwargs = {
        **inputs,
        "max_new_tokens": config.reflection_max_new_tokens,
        "temperature": config.reflection_temperature,
    }
    if config.reflection_temperature and config.reflection_temperature > 0:
        generation_kwargs["do_sample"] = True

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    logger.log("Running LLM reflection")
    generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    match = re.search(r"(<reflection>(.*?)</reflection>)", generated_text, re.DOTALL)
    reflection_text = match.group(1).strip() if match else generated_text.strip()
    logger.log(f"Reflection: {reflection_text}")
    return reflection_text


def llm_intervention(
    query: str,
    step_summary: str,
    reflection: str,
    feedback_llm,
    logger: TraceLogger,
    openai_usage_acc: dict[str, float],
):
    from langchain_community.callbacks import get_openai_callback

    from hitl.prompts import build_reflection_feedback_prompt
    from hitl.schemas import ReflectionFeedbackOutput, reflection_feedback_schema

    prompt = build_reflection_feedback_prompt(
        query=query,
        step_summary=step_summary,
        reflection=reflection,
        schema=reflection_feedback_schema(),
    )

    with get_openai_callback() as callback:
        response = feedback_llm.with_structured_output(ReflectionFeedbackOutput).invoke(prompt)
        openai_usage_acc["total_tokens"] += callback.total_tokens
        openai_usage_acc["total_cost"] += getattr(callback, "total_cost", 0.0) or 0.0

    logger.log(f"correct_reflection: {response.correct_reflection}")
    logger.log(f"need_correction: {response.need_correction}")
    logger.log(f"unclear_step: {response.unclear_step}")
    logger.log(f"llm_feedback: {response.feedback}")
    return response


def llm_correction(
    query: str,
    generated_steps: list[str],
    user_feedback,
    model,
    tokenizer,
    logger: TraceLogger,
    config: ExperimentConfig,
) -> str:
    from hitl.generation import generate_step
    from hitl.prompts import build_correction_prompt

    feedback_text = user_feedback if isinstance(user_feedback, str) else user_feedback.feedback
    correction_prompt = build_correction_prompt(query, generated_steps, feedback_text)
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
    from hitl.prompts import build_initial_prompt, build_step_summary
    from hitl.rewards import generate_reward

    if logger is None:
        logger = TraceLogger()

    full_prompt = build_initial_prompt(query)
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
        step_reward = generate_reward(generated_steps, query, reward_model, reward_tokenizer)
        time_tracking["generate_reward_time"] += time.time() - start_time
        last_score = _latest_score(step_reward)
        logger.log(f"Step {len(generated_steps)} Reward: {last_score:.3f}")

        correction_attempts = 0
        while (
            last_score <= config.reward_threshold
            and correction_attempts < config.max_corrections
        ):
            correction_attempts += 1
            logger.log("Low reward detected. Triggering reflection.")
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
            llm_feedback = llm_intervention(
                query,
                step_summary,
                reflection,
                feedback_llm,
                logger,
                openai_usage_acc,
            )
            time_tracking["intervention_time"] += time.time() - start_time

            if llm_feedback.need_correction:
                unclear_step = llm_feedback.unclear_step
                generated_steps = generated_steps[:unclear_step]
                last_step_scores = last_step_scores[: unclear_step - 1] if unclear_step > 1 else []

            if not llm_feedback.need_correction:
                last_score = 0.9
                break

            start_time = time.time()
            corrected = llm_correction(
                query,
                generated_steps,
                llm_feedback,
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

        last_step_scores.append(float(last_score))
        full_prompt += new_step + "\n"

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
    }
    result.update({k: round(v, 3) for k, v in time_tracking.items()})
    return result


def _validate_config(config: ExperimentConfig) -> None:
    if config.input_path is None:
        raise ValueError("--input is required.")
    if config.output_path is None and not config.dry_run:
        raise ValueError("--output is required unless --dry-run is set.")


def run(config: ExperimentConfig) -> None:
    _validate_config(config)

    dataset = list(read_jsonl(config.input_path))
    if config.limit is not None:
        dataset = dataset[: config.limit]

    print(f"Loaded {len(dataset)} item(s) from {config.input_path}")
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
