"""Structured outputs used by feedback models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeedbackOutput(BaseModel):
    need_correction: bool = Field(..., description="Whether a correction is needed.")
    unclear_step: int = Field(..., description="Step number that needs to be fixed.")
    feedback: str = Field(..., description="Simple and direct hint for correction.")


class ReflectionFeedbackOutput(BaseModel):
    correct_reflection: bool = Field(
        ...,
        description="Whether the reflection is correct and helpful for feedback.",
    )
    need_correction: bool = Field(..., description="Whether a correction is needed.")
    unclear_step: int = Field(..., description="Step number that needs to be fixed.")
    feedback: str = Field(..., description="Simple and direct hint for correction.")


class GlobalDirectFeedbackOutput(BaseModel):
    need_correction: bool = Field(..., description="Whether a correction is needed.")
    unclear_step: int = Field(..., description="Step number that needs to be fixed.")
    corrected_step: str = Field(
        ...,
        description="Corrected step text including <step> tags.",
    )
    reason: str = Field(..., description="Reason for the correction.")
    global_advice: str = Field(
        ...,
        description="High-level advice for the remaining solution. Empty if no update is needed.",
    )


class GlobalSoftFeedbackOutput(BaseModel):
    need_correction: bool = Field(..., description="Whether a correction is needed.")
    unclear_step: int = Field(..., description="Step number that needs to be fixed.")
    feedback: str = Field(..., description="Local feedback or hint for the specific step.")
    global_advice: str = Field(
        ...,
        description="High-level advice for the remaining solution. Empty if no update is needed.",
    )


class GPQAFeedbackOutput(BaseModel):
    need_correction: bool = Field(..., description="Whether a correction is needed.")
    unclear_step: int = Field(..., description="Step number that needs to be fixed.")
    corrected_step: str = Field(..., description="Corrected step text.")
    reason: str = Field(..., description="Reason for the correction.")


class TrajectoryRewardOutput(BaseModel):
    reasoning: str = Field(
        ...,
        description="Reasoning about the correctness of the solution trajectory so far.",
    )
    score: float = Field(
        ...,
        description="Score between 0.0 and 1.0. 1.0 means on track; 0.0 means wrong.",
    )


def _schema_for(model_class: type[BaseModel]) -> dict:
    """Return a Pydantic schema across v1/v2."""
    if hasattr(model_class, "model_json_schema"):
        return model_class.model_json_schema()
    return model_class.schema()


def feedback_schema() -> dict:
    return _schema_for(FeedbackOutput)


def reflection_feedback_schema() -> dict:
    return _schema_for(ReflectionFeedbackOutput)


def global_direct_feedback_schema() -> dict:
    return _schema_for(GlobalDirectFeedbackOutput)


def global_soft_feedback_schema() -> dict:
    return _schema_for(GlobalSoftFeedbackOutput)


def gpqa_feedback_schema() -> dict:
    return _schema_for(GPQAFeedbackOutput)


def trajectory_reward_schema() -> dict:
    return _schema_for(TrajectoryRewardOutput)
