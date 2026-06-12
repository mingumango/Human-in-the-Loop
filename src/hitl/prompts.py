"""Prompt builders shared by experiment strategies."""

from __future__ import annotations

from typing import Any


def build_initial_prompt(query: str) -> str:
    return f"""
You are a helpful math tutor. solve the math problem.
Show each step in the format: <step> ... </step>
example)
Question:
what is 13 times 4?

Solution:
<step> First, we need to understand the problem. We are asked to find the product of 13 and 4 </step>
<step> Next, we can define our variables. Let x be the product of 13 and 4 </step>
<step> Now, we can formulate the equation: x = 13 * 4 </step>
<step> Then, we solve the equation: x = 52</step>
<step> Finally, we report the answer: The product of 13 and 4 is 52 </step>
<answer> 52 </answer>

solve this question below
Question:
{query}
Solution:
""".strip()


def build_prompt_with_context(query: str, global_advice: str = "") -> str:
    base_prompt = build_initial_prompt(query)
    if not global_advice:
        return base_prompt

    advice_section = f"""
<strategic_advice>
IMPORTANT: Keep the following advice in mind for the entire solution:
{global_advice}
</strategic_advice>
""".strip()
    return base_prompt.replace("solve this question below", advice_section + "\n\nsolve this question below")


def build_step_summary(generated_steps: list[str], step_reward: list[list[float]]) -> str:
    summary_lines = []
    scores = step_reward[0] if step_reward else []
    for i, step in enumerate(generated_steps):
        avg_reward = scores[i] if i < len(scores) else 0.0
        summary_lines.append(f"[Step {i + 1}] {step.strip()}\nReward: {avg_reward:.3f}")
    return "\n".join(summary_lines)


def build_feedback_prompt(query: str, step_summary: str, schema: dict[str, Any]) -> str:
    return f"""
You are an expert math tutor acting as a "Human-in-the-Loop" assistant.
A student AI is trying to solve a math problem and got stuck.
You need to provide a concise hint to help them fix their reasoning.

Here is the original problem:
{query}

Here are the student's reasoning steps so far (with rewards):
{step_summary}

provide a single, concise piece of feedback or a guiding question
to help them correct their mistake.

Your feedback should be simple and direct, like a human tutor would type it.
Do not solve the entire problem. Just address the particular step.
If student's step has no problem, return False in need_correction and empty string in feedback.
example)
"need_correction": True
"unclear_step": 3
"feedback": "No you don't need to multiply 34 again. Kelly's total distance equation is 200 + 2*40*34, that's all."

Your response must be a JSON object matching the following Pydantic schema:
{schema}
""".strip()


def build_reflection_prompt(query: str, step_summary: str) -> str:
    return f"""
You are a reflective AI reasoning agent.
You have been solving the following math problem:

Question:
{query}

Here are your reasoning steps and their corresponding rewards:
{step_summary}

Now, reflect on your reasoning process.

<instruction>
- Provide a detailed explanation of your understanding of the problem and the steps you took to solve it based on your reasoning steps.
- Identify which step was initially wrong and caused the problem. It may not be the last step. Think carefully.
- Explain *why* you think that step may be problematic.
- Describe what kind of information, hint, or feedback from a human could help you improve.
- Keep your reflection concise and simple.
</instruction>

Your response should be structured as:
<reflection>
1. [State of Understanding]: ...
2. [Possible Reason for Confusion]: ...
3. [Help Needed From Human]: ...
</reflection>
""".strip()


def build_reflection_feedback_prompt(
    query: str,
    step_summary: str,
    reflection: str,
    schema: dict[str, Any],
) -> str:
    return f"""
You are an expert math tutor acting as a "Human-in-the-Loop" assistant.
A student AI is trying to solve a math problem and got stuck.
You need to provide a concise hint to help them fix their reasoning.

Here is the original problem:
{query}

Here are the student's reasoning steps so far (with rewards):
{step_summary}

Here is the student's own "reflection" on what went wrong.

{reflection}

Based on the student's reflection, especially the unclear step and possible reasons,
provide a single, concise piece of feedback or a guiding question
to help them correct their mistake.

Your feedback should be simple and direct, like a human tutor would type it.
Do not solve the entire problem. Just address the particular step.
If student's step has no problem, return False in need_correction and empty string in feedback.
example)
"correct_reflection": True
"need_correction": True
"unclear_step": 3
"feedback": "No you don't need to multiply 34 again. Kelly's total distance equation is 200 + 2*40*34, that's all."

Your response must be a JSON object matching the following Pydantic schema:
{schema}
""".strip()


def build_correction_prompt(
    query: str,
    generated_steps: list[str],
    feedback_text: str,
) -> str:
    previous_steps = "\n".join(generated_steps[:-1])
    all_steps = "\n".join(generated_steps)

    return f"""
You previously generated the following reasoning steps for the math problem. But got wrong on the last step
What you have generated so far:
{all_steps}

A human provided feedback on last step to help you improve:
{feedback_text}

Now, write the next reasoning step (<step>...</step>) to reflect this feedback accurately.

You are a helpful math tutor. solve the math problem.
Show each step in the format: <step> ... </step>
example)
Question:
what is 13 times 4?

Solution:
<step> First, we need to understand the problem. We are asked to find the product of 13 and 4 </step>
<step> Next, we can define our variables. Let x be the product of 13 and 4 </step>
<step> Now, we can formulate the equation: x = 13 * 4 </step>
<step> Then, we solve the equation: x = 52</step>
<step> Finally, we report the answer: The product of 13 and 4 is 52 </step>
<answer> 52 </answer>

solve this question below
Question:
{query}

A human provided feedback to help you improve:
\"\"\"{feedback_text}\"\"\"

Solution:
{previous_steps}
""".strip()


def build_global_direct_feedback_prompt(
    query: str,
    step_summary: str,
    reflection: str,
    schema: dict[str, Any],
) -> str:
    return f"""
You are an expert math tutor acting as a "Human-in-the-Loop" assistant.
A student AI is trying to solve a math problem and got stuck.
You need to fix the wrong step for the student AI if there is any.

Here is the original problem:
{query}

Here are the student's reasoning steps so far (with rewards):
{step_summary}

Here is the student's own "reflection" on what went wrong:
{reflection}

Your Task:
1. Local Correction: If student's reasoning is incorrect, identify the step where the mistake begins and provide a corrected version of the step.
2. Global Advice: Provide a high-level strategy or reminder to guide the student for the entire remaining problem. This advice will be shown to the student continuously.

There might be no problem in reasoning at all. If student's step has no problem, return False in need_correction and empty string in corrected_step and reason.

Example:
"need_correction": True,
"unclear_step": 3,
"corrected_step": "<step> Since he bought 20 more corns, new total corns = 80 + 20 = 100 </step>",
"reason": "You should add the extra corns, not subtract them.",
"global_advice": "Pay close attention to keywords like more or less. After finding the new total, remember the remaining operation."

Your response must be a JSON object matching the following Pydantic schema:
{schema}
""".strip()


def build_global_soft_feedback_prompt(
    query: str,
    step_summary: str,
    reflection: str,
    schema: dict[str, Any],
) -> str:
    return f"""
You are an expert math tutor acting as a "Human-in-the-Loop" assistant.
A student AI is trying to solve a math problem and got stuck.

Here is the original problem:
{query}

Here are the student's reasoning steps so far (with rewards):
{step_summary}

Here is the student's own "reflection" on what went wrong:
{reflection}

Your Task:
1. Local Feedback: Provide a concise hint to correct the specific wrong step, if any.
2. Global Advice: Provide a high-level strategy or reminder to guide the student for the entire remaining problem. This advice will be shown to the student continuously.

There might be no problem in reasoning at all. If student's step has no problem, return False in need_correction and empty string in feedback.

Example:
"need_correction": True,
"unclear_step": 3,
"feedback": "You multiplied by 12, but there are only 10 months relevant here.",
"global_advice": "Remember to check whether the unit is months or years before calculating the total."

Your response must be a JSON object matching the following Pydantic schema:
{schema}
""".strip()


def build_global_direct_correction_prompt(
    query: str,
    generated_steps: list[str],
    corrected_step: str,
    reason: str = "",
    global_advice: str = "",
) -> str:
    previous_steps = "\n".join(generated_steps[:-1])
    feedback_lines = [f"corrected_step: {corrected_step}"]
    if reason:
        feedback_lines.append(f"reason: {reason}")
    if global_advice:
        feedback_lines.append(f"Overall Strategy: {global_advice}")
    feedback_text = "\n".join(feedback_lines)

    return f"""
You previously generated the following reasoning steps for the math problem. But got wrong on the last step.
What you have generated so far:
{"\n".join(generated_steps)}

A human provided the following feedback:
\"\"\"{feedback_text}\"\"\"

Now, fix the last step with the corrected step that human provided for you.

You are a helpful math tutor. solve the math problem.
Show each step in the format: <step> ... </step>
example)
Question:
what is 13 times 4?

Solution:
<step> First, we need to understand the problem. We are asked to find the product of 13 and 4 </step>
<step> Next, we can define our variables. Let x be the product of 13 and 4 </step>
<step> Now, we can formulate the equation: x = 13 * 4 </step>
<step> Then, we solve the equation: x = 52</step>
<step> Finally, we report the answer: The product of 13 and 4 is 52 </step>
<answer> 52 </answer>

solve this question below
Question:
{query}

Solution:
{previous_steps}
""".strip()


def build_global_soft_correction_prompt(
    query: str,
    generated_steps: list[str],
    feedback: str,
    global_advice: str = "",
) -> str:
    previous_steps = "\n".join(generated_steps[:-1])
    combined_feedback = f"Specific Hint: {feedback}"
    if global_advice:
        combined_feedback += f"\nOverall Strategy: {global_advice}"

    return f"""
You previously generated the following reasoning steps for the math problem. But got wrong on the last step.
What you have generated so far:
{"\n".join(generated_steps)}

A human provided the following feedback:
\"\"\"{combined_feedback}\"\"\"

Now, write the next reasoning step (<step>...</step>) to reflect this feedback accurately.

You are a helpful math tutor. solve the math problem.
Show each step in the format: <step> ... </step>
example)
Question:
what is 13 times 4?

Solution:
<step> First, we need to understand the problem. We are asked to find the product of 13 and 4 </step>
<step> Next, we can define our variables. Let x be the product of 13 and 4 </step>
<step> Now, we can formulate the equation: x = 13 * 4 </step>
<step> Then, we solve the equation: x = 52</step>
<step> Finally, we report the answer: The product of 13 and 4 is 52 </step>
<answer> 52 </answer>

solve this question below
Question:
{query}

Solution:
{previous_steps}
""".strip()


def build_gpqa_step_summary(
    generated_steps: list[str],
    past_scores: list[float],
    current_score: float,
) -> str:
    lines = []
    total_steps = len(generated_steps)
    for i, step in enumerate(generated_steps):
        if i == total_steps - 1:
            reward = current_score
        else:
            reward = past_scores[i] if i < len(past_scores) else 0.0
        lines.append(
            f"[Step {i + 1}] {step.strip()}\n"
            f"(Trajectory Score at this point: {reward:.3f})"
        )
    return "\n".join(lines)


def build_gpqa_trajectory_reward_prompt(
    query: str,
    options: dict[str, str],
    generated_steps: list[str],
) -> str:
    full_solution = "\n".join(generated_steps)
    return f"""
You are a strict scientific reasoning grader. Your task is to evaluate the correctness,
soundness, and scientific validity of the solution trajectory generated so far
for a GPQA Diamond multiple-choice question.

The question is high-difficulty (PhD-level), and your evaluation should reflect whether the reasoning
is logically consistent, scientifically valid, and moving toward the correct conclusion.

Problem:
{query}

Options:
A) {options.get("A", "")}
B) {options.get("B", "")}
C) {options.get("C", "")}
D) {options.get("D", "")}

Solution Generated So Far:
{full_solution}

Assess whether the solution is logically correct and heading towards the right answer.

<Grading Criteria>
1. Partial Evaluation: The solution is generated step-by-step. Do not penalize for being incomplete or not having reached the final answer yet.
2. Scientific Validity: Check whether the reasoning uses scientifically sound principles, facts, and logic.
3. Relevance & Focus: The reasoning should be relevant to solving the given GPQA question.
4. Heading Toward Correct Answer: If the reasoning is generally progressing toward the correct conclusion, assign a higher score.
5. Intervention Threshold: The human intervention occurs if the score is low. Review your score accordingly.
</Grading Criteria>

Return a single JSON object with reasoning and score.
""".strip()


def build_gpqa_feedback_prompt(
    query: str,
    options: dict[str, str],
    step_summary: str,
    schema: dict[str, Any],
) -> str:
    return f"""
You are an expert scientific reasoning tutor acting as a "Human-in-the-Loop" assistant.
A student AI is trying to solve a GPQA Diamond high-difficulty science question.
Their reasoning may contain scientific mistakes, logical gaps, or incorrect assumptions.

Your job is to provide a concise hint that corrects only the specific step where the reasoning first becomes incorrect.

Here is the original problem:
{query}

Options:
A) {options.get("A", "")}
B) {options.get("B", "")}
C) {options.get("C", "")}
D) {options.get("D", "")}

Here are the student's reasoning steps so far:
{step_summary}

If the student's reasoning is incorrect, identify the step where the mistake begins and provide a corrected version of the step.

Do not solve the entire problem. Just address the particular step.
If the student's step has no problem, return False in need_correction and empty strings in corrected_step and reason.

Your response must be a JSON object matching the following Pydantic schema:
{schema}
""".strip()


def build_gpqa_correction_prompt(
    query: str,
    options: dict[str, str],
    generated_steps: list[str],
    corrected_step: str,
) -> str:
    previous_steps = "\n".join(generated_steps[:-1])
    all_steps = "\n".join(generated_steps)
    return f"""
You previously generated the following reasoning steps for the GPQA science problem. But got wrong on the last step.
What you have generated so far:
{all_steps}

A human provided feedback on last step to help you improve:
{corrected_step}

Now, fix the last step with the corrected step that human provided for you.

You are a helpful science tutor. You will be given a multiple-choice question.
Think step by step and show your reasoning inside <step>...</step> tags.
At the end, choose exactly one option among A, B, C, and D.

The last line of your answer MUST be in the format:
<answer> LETTER </answer>
where LETTER is one of A, B, C, or D.

Question:
{query}

Options:
A) {options.get("A", "")}
B) {options.get("B", "")}
C) {options.get("C", "")}
D) {options.get("D", "")}

A human provided feedback to help you improve:
\"\"\"{corrected_step}\"\"\"

Solution:
{previous_steps}
""".strip()
