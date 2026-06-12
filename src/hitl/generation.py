"""Step generation helpers."""

from __future__ import annotations

import re
from typing import Optional

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnStep(StoppingCriteria):
    def __init__(self, stop_strs: list[str], tokenizer, prompt_length: int = 0) -> None:
        super().__init__()
        self.stop_strs = stop_strs
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_ids = input_ids[:, self.prompt_length :]
        if generated_ids.shape[1] == 0:
            return False
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return any(stop in generated_text for stop in self.stop_strs)


def generate_step(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 150,
    temperature: float = 0.5,
    token_usage_acc: dict[str, float] | None = None,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    if token_usage_acc is not None:
        token_usage_acc["llama_input_tokens"] += prompt_length

    stop_criteria = StopOnStep(
        stop_strs=["</step>", "</answer>"],
        tokenizer=tokenizer,
        prompt_length=prompt_length,
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop_criteria]),
    }
    if temperature and temperature > 0:
        generation_kwargs["do_sample"] = True

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    if token_usage_acc is not None:
        token_usage_acc["llama_output_tokens"] += outputs.shape[1] - prompt_length

    new_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    step_match = re.search(r"(<step>.*?</step>)", new_text, flags=re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"(<answer>.*?</answer>)", new_text, flags=re.DOTALL | re.IGNORECASE)
    if step_match:
        return step_match.group(1)
    if answer_match:
        return answer_match.group(1)
    return new_text.strip()


def generate_until_answer(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 3000,
    temperature: float = 0.7,
    token_usage_acc: dict[str, float] | None = None,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    if token_usage_acc is not None:
        token_usage_acc["llama_input_tokens"] += prompt_length

    stop_criteria = StopOnStep(
        stop_strs=["</answer>"],
        tokenizer=tokenizer,
        prompt_length=prompt_length,
    )
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop_criteria]),
    }
    if temperature and temperature > 0:
        generation_kwargs["do_sample"] = True

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    if token_usage_acc is not None:
        token_usage_acc["llama_output_tokens"] += outputs.shape[1] - prompt_length

    return tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)


def extract_final_answer_text(step_texts: list[str]) -> Optional[str]:
    joined = "\n".join(step_texts)
    match = re.search(r"<answer>(.*?)</answer>", joined, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
