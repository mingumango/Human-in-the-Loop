"""Reward-model scoring helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor) -> list[list[float]]:
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)

    all_scores = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        all_scores.append(positive_probs.detach().cpu().tolist())
    return all_scores


def generate_reward(
    generated_steps: list[str],
    query: str,
    reward_model,
    reward_tokenizer,
    token_usage_acc: dict[str, float] | None = None,
) -> list[list[float]]:
    if not generated_steps:
        return [[0.0]]

    full_response = "<extra_0>".join(generated_steps) + "<extra_0>"
    messages = [
        {"role": "system", "content": "Solve the question with reasoning steps."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": full_response},
    ]
    conversation = reward_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    input_ids = reward_tokenizer.encode(conversation, return_tensors="pt").to(reward_model.device)
    if token_usage_acc is not None:
        token_usage_acc["qwen_input_tokens"] += input_ids.shape[1]

    with torch.no_grad():
        outputs = reward_model(input_ids=input_ids, use_cache=False)

    step_sep_id = reward_tokenizer.encode("<extra_0>")[0]
    token_masks = input_ids == step_sep_id
    return make_step_rewards(outputs[0], token_masks)
