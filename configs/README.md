# Experiment Configs

Store reproducible experiment settings here. Prefer one config per named run,
with paths relative to the repository root when possible.

Example shape:

```yaml
dataset: gsm8k
strategy: reflection
input: data/processed/gsm8k_test_common_filtered.jsonl
output: outputs/runs/gsm8k_reflection_llama8b.jsonl
model: meta-llama/Llama-3.1-8B-Instruct
reward_model: Qwen/Qwen2.5-Math-PRM-7B
max_steps: 17
resume: true
```
