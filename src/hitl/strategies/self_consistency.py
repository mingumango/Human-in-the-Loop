"""Self-consistency HITL strategy.

This migrates ``archive/legacy_scripts/run_self_consistency_hitl.py``. It
keeps multiple sampled reasoning paths alive, pauses paths with low PRM reward,
and triggers HITL when all active paths are stuck. The corrected best path is
then replicated to all samples before sampling resumes.
"""

from __future__ import annotations

import time
from typing import Any

from tqdm import tqdm

from hitl.config import ExperimentConfig
from hitl.evaluation import numeric_answer_correct
from hitl.io import append_jsonl, existing_indices, read_jsonl
from hitl.trace import TraceLogger


def _init_token_usage() -> dict[str, float]:
    return {
        "llama_input_tokens": 0,
        "llama_output_tokens": 0,
        "qwen_input_tokens": 0,
        "qwen_output_tokens": 0,
    }


def _step_summary(generated_steps: list[str], step_rewards: list[float]) -> str:
    lines = []
    for i, step in enumerate(generated_steps):
        reward = step_rewards[i] if i < len(step_rewards) else 0.0
        lines.append(f"[Step {i + 1}] {step.strip()}\nReward: {reward:.3f}")
    return "\n".join(lines)


def _best_finished_or_longest(paths: list[list[str]], path_scores: list[list[float]], finished: list[bool]) -> int:
    finished_indices = [i for i, is_finished in enumerate(finished) if is_finished]
    if finished_indices:
        best_idx = finished_indices[0]
        best_avg_reward = -1.0
        for idx in finished_indices:
            scores = path_scores[idx]
            avg_reward = sum(scores) / len(scores) if scores else 0.0
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_idx = idx
        return best_idx

    return max(range(len(paths)), key=lambda i: len(paths[i]))


def run_single_question(
    query: str,
    model,
    tokenizer,
    reward_model,
    reward_tokenizer,
    feedback_llm,
    config: ExperimentConfig,
    logger: TraceLogger | None = None,
    token_usage_acc: dict[str, float] | None = None,
) -> dict[str, Any]:
    from hitl.generation import extract_final_answer_text, generate_step
    from hitl.prompts import build_initial_prompt
    from hitl.rewards import generate_reward
    from hitl.strategies.reflection import llm_correction, llm_intervention, llm_reflection

    if logger is None:
        logger = TraceLogger()
    if token_usage_acc is None:
        token_usage_acc = _init_token_usage()

    openai_usage_acc = {"total_tokens": 0, "total_cost": 0.0}
    num_samples = config.num_samples

    paths: list[list[str]] = [[] for _ in range(num_samples)]
    path_scores: list[list[float]] = [[] for _ in range(num_samples)]
    prompts = [build_initial_prompt(query) for _ in range(num_samples)]
    finished = [False] * num_samples
    paused = [False] * num_samples
    paused_states: list[dict[str, Any] | None] = [None] * num_samples

    logger.log(f"--- Starting self-consistency HITL (N={num_samples}) ---")
    main_loop_limit = config.max_steps * 2

    for _ in range(main_loop_limit):
        if all(finished):
            logger.log("All samples finished naturally.")
            break

        active_indices = [
            i for i in range(num_samples) if not finished[i] and not paused[i]
        ]

        for sample_idx in active_indices:
            new_step = generate_step(
                prompts[sample_idx],
                model,
                tokenizer,
                max_new_tokens=config.generation_max_new_tokens,
                temperature=config.generation_temperature,
                token_usage_acc=token_usage_acc,
            )
            temp_path = paths[sample_idx] + [new_step]
            rewards_obj = generate_reward(
                temp_path,
                query,
                reward_model,
                reward_tokenizer,
                token_usage_acc=token_usage_acc,
            )
            last_score = rewards_obj[0][-1] if rewards_obj and rewards_obj[0] else 0.0

            if last_score < config.reward_threshold:
                paused[sample_idx] = True
                paused_states[sample_idx] = {"step": new_step, "score": last_score}
                logger.log(
                    f"[Sample {sample_idx}] Paused. Reward {last_score:.3f} < "
                    f"{config.reward_threshold}."
                )
                continue

            paths[sample_idx].append(new_step)
            path_scores[sample_idx].append(last_score)
            prompts[sample_idx] += new_step + "\n"

            if "</answer>" in new_step.lower():
                finished[sample_idx] = True
                logger.log(f"[Sample {sample_idx}] Finished. Final score: {last_score:.3f}")

        remaining_indices = [i for i in range(num_samples) if not finished[i]]

        if any(finished):
            if remaining_indices and all(paused[i] for i in remaining_indices):
                logger.log("Some samples finished and the rest are stuck. Stopping early.")
                break
            continue

        if remaining_indices and all(paused[i] for i in remaining_indices):
            logger.log("Intervention triggered: all active paths are paused.")

            best_idx = -1
            best_min_reward = -1.0
            for idx in remaining_indices:
                paused_state = paused_states[idx] or {"score": 0.0, "step": ""}
                history_scores = path_scores[idx] + [paused_state["score"]]
                min_reward = min(history_scores) if history_scores else 0.0
                if min_reward > best_min_reward:
                    best_min_reward = min_reward
                    best_idx = idx

            logger.log(f"Selected Sample {best_idx} for correction (maximin={best_min_reward:.3f}).")

            paused_state = paused_states[best_idx] or {"score": 0.0, "step": ""}
            target_path = paths[best_idx] + [paused_state["step"]]
            target_rewards = path_scores[best_idx] + [paused_state["score"]]

            step_summary = _step_summary(target_path, target_rewards)
            reflection = llm_reflection(
                target_path,
                [target_rewards],
                query,
                model,
                tokenizer,
                logger,
                config,
            )
            feedback = llm_intervention(
                query,
                step_summary,
                reflection,
                feedback_llm,
                logger,
                openai_usage_acc,
            )

            if feedback.need_correction:
                best_correction = ""
                best_correction_score = -1.0
                logger.log(f"Attempting correction up to {config.max_corrections} time(s).")

                for attempt in range(config.max_corrections):
                    corrected_step = llm_correction(
                        query,
                        target_path,
                        feedback,
                        model,
                        tokenizer,
                        logger,
                        config,
                    )
                    temp_full_path = paths[best_idx] + [corrected_step]
                    new_rewards = generate_reward(
                        temp_full_path,
                        query,
                        reward_model,
                        reward_tokenizer,
                        token_usage_acc=token_usage_acc,
                    )
                    correction_score = (
                        new_rewards[0][-1] if new_rewards and new_rewards[0] else 0.0
                    )
                    logger.log(f"[Correction try {attempt + 1}] Reward: {correction_score:.3f}")

                    if correction_score > best_correction_score:
                        best_correction_score = correction_score
                        best_correction = corrected_step

                    if correction_score >= config.reward_threshold:
                        logger.log("Threshold passed. Accepting correction.")
                        break

                final_step_text = best_correction
                final_step_score = best_correction_score
            else:
                logger.log("No correction needed. Forcing reward to 0.9.")
                final_step_text = paused_state["step"]
                final_step_score = 0.9

            paths[best_idx].append(final_step_text)
            path_scores[best_idx].append(final_step_score)
            prompts[best_idx] += final_step_text + "\n"
            is_finished = "</answer>" in final_step_text.lower()

            logger.log("Sync: replicating corrected state to all samples.")
            base_path = list(paths[best_idx])
            base_scores = list(path_scores[best_idx])
            base_prompt = prompts[best_idx]

            for sample_idx in range(num_samples):
                paths[sample_idx] = list(base_path)
                path_scores[sample_idx] = list(base_scores)
                prompts[sample_idx] = base_prompt
                paused[sample_idx] = False
                paused_states[sample_idx] = None
                finished[sample_idx] = is_finished

    candidates = []
    for sample_idx in range(num_samples):
        answer_text = extract_final_answer_text(paths[sample_idx]) or ""
        candidates.append(
            {
                "sample_id": sample_idx,
                "generated_steps": paths[sample_idx],
                "generated_rewards": path_scores[sample_idx],
                "model_answer": answer_text,
                "finished": finished[sample_idx],
            }
        )

    best_idx = _best_finished_or_longest(paths, path_scores, finished)

    return {
        "candidates": candidates,
        "best_idx": best_idx,
        "trace": logger.dump(),
        "openai_total_tokens": int(openai_usage_acc["total_tokens"]),
        "openai_total_cost_usd": float(openai_usage_acc["total_cost"]),
        "llama_input_tokens": int(token_usage_acc["llama_input_tokens"]),
        "llama_output_tokens": int(token_usage_acc["llama_output_tokens"]),
        "qwen_input_tokens": int(token_usage_acc["qwen_input_tokens"]),
        "qwen_output_tokens": int(token_usage_acc.get("qwen_output_tokens", 0)),
    }


def _validate_config(config: ExperimentConfig) -> None:
    if config.input_path is None:
        raise ValueError("--input is required.")
    if config.output_path is None and not config.dry_run:
        raise ValueError("--output is required unless --dry-run is set.")
    if config.num_samples < 1:
        raise ValueError("--num-samples must be >= 1.")


def run(config: ExperimentConfig) -> None:
    _validate_config(config)

    dataset = list(read_jsonl(config.input_path))
    if config.limit is not None:
        dataset = dataset[: config.limit]

    print(f"Loaded {len(dataset)} item(s) from {config.input_path}")
    print(f"Self-consistency samples: {config.num_samples}")
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
    token_usage_acc = _init_token_usage()

    for obj in tqdm(dataset, desc="Questions"):
        idx = obj.get("index")
        if config.resume and idx in already_done:
            continue

        start_time = time.time()
        result = run_single_question(
            query=obj.get("question", ""),
            model=model,
            tokenizer=tokenizer,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            feedback_llm=feedback_llm,
            config=config,
            logger=TraceLogger(),
            token_usage_acc=token_usage_acc,
        )
        duration_seconds = time.time() - start_time

        best_candidate = result["candidates"][result["best_idx"]]
        best_model_answer = best_candidate["model_answer"]
        gt_final = obj.get("ground_truth_final")

        out_item = {
            "index": idx,
            "question": obj.get("question"),
            "ground_truth_text": obj.get("ground_truth_text", ""),
            "ground_truth_final": gt_final,
            "candidates": result["candidates"],
            "selected_sample_index": result["best_idx"],
            "best_model_answer": best_model_answer,
            "correct": numeric_answer_correct(best_model_answer, gt_final),
            "trace": result["trace"],
            "openai_total_tokens": result["openai_total_tokens"],
            "openai_total_cost_usd": result["openai_total_cost_usd"],
            "llama_input_tokens": result["llama_input_tokens"],
            "llama_output_tokens": result["llama_output_tokens"],
            "qwen_input_tokens": result["qwen_input_tokens"],
            "qwen_output_tokens": result["qwen_output_tokens"],
            "processing_time_seconds": round(duration_seconds, 2),
        }
        append_jsonl(config.output_path, out_item)

    print(f"Done. Results written to: {config.output_path}")
