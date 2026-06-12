"""Model loading helpers."""

from __future__ import annotations

import os

import torch

from .config import ExperimentConfig


def _torch_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def load_reasoning_model(config: ExperimentConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.model is None:
        raise ValueError("--model is required.")

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model_kwargs = {"device_map": "auto"}
    if config.reasoning_load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    else:
        model_kwargs["torch_dtype"] = _torch_dtype(config.reasoning_dtype)

    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_reward_model(config: ExperimentConfig):
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    if config.reward_model is None:
        raise ValueError("--reward-model is required.")

    quantization_config = None
    if config.reward_load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(config.reward_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        config.reward_model,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


def build_feedback_llm(config: ExperimentConfig):
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    return ChatOpenAI(
        model=config.openai_model,
        temperature=config.openai_temperature,
        api_key=api_key,
        max_tokens=config.openai_max_tokens,
    )
