"""Evaluation helpers for generated answers."""

from __future__ import annotations

import math
import re
from typing import Optional


def extract_number(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if match:
        return float(match.group(0))
    return None


def numeric_answer_correct(model_answer: str, ground_truth_final) -> bool:
    model_answer_num = extract_number(model_answer)
    if ground_truth_final is None or model_answer_num is None:
        return False
    try:
        return math.isclose(float(ground_truth_final), model_answer_num, rel_tol=1e-9, abs_tol=1e-9)
    except Exception:
        return False


def extract_choice_letter(text: str) -> str | None:
    match = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([ABCD])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def choice_answer_correct(model_answer: str, ground_truth_final) -> bool:
    predicted = extract_choice_letter(model_answer)
    if predicted is None or ground_truth_final is None:
        return False
    return predicted == str(ground_truth_final).strip().upper()
