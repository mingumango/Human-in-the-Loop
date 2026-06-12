# HITL Experiments

Reusable runners for human-in-the-loop and model-feedback experiments on
step-by-step reasoning tasks. The current supported datasets are GSM8K-style
math JSONL files and GPQA Diamond multiple-choice data.

The old standalone research scripts can be kept locally under `archive/` for
reference, but that directory is ignored by git. New work should go through
`src/hitl/` and `configs/`.

## Repository Layout

```text
src/hitl/                 Python package and CLI entry point
src/hitl/strategies/      Experiment strategies
src/hitl/datasets/        Dataset adapters and prompt/evaluation glue
src/hitl/router/          Placeholder for router work
configs/                  Reproducible YAML experiment configs
data/                     Local datasets, ignored by git
outputs/                  Run outputs, ignored by git
archive/                  Local pre-migration reference, ignored by git
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If the package is not installed yet, use:

```bash
PYTHONPATH=src python -m hitl.run_experiments --help
```

After `pip install -e .`, use:

```bash
hitl-run --help
```

Set `OPENAI_API_KEY` for strategies that use OpenAI feedback or reward models:

```bash
export OPENAI_API_KEY=...
```

## Quick Validation

`--dry-run` validates config loading, dataset loading, and dispatch without
loading local models or OpenAI clients.

```bash
hitl-run --config configs/gsm8k_no_reflection_llama8b.example.yaml --dry-run --limit 1
hitl-run --config configs/gpqa_baseline_llama8b.example.yaml --dry-run --limit 1
```

Run a syntax check without importing heavy model dependencies:

```bash
python -m py_compile src/hitl/*.py src/hitl/strategies/*.py src/hitl/datasets/*.py src/hitl/router/*.py
```

## Datasets

`gsm8k` expects JSONL rows with fields such as:

```json
{
  "index": 12,
  "question": "...",
  "ground_truth_text": "...",
  "ground_truth_final": "13"
}
```

`gpqa` supports either JSONL rows or a Hugging Face dataset saved to disk. The
current example config points to:

```text
data/external/gpqa_diamond_mc_shuffled
```

The GPQA adapter reads this with `datasets.load_from_disk` when available and
falls back to `pyarrow` for local `.arrow` files.

The example configs use `data/` paths. In this workspace those paths are local
symlinks back to `archive/`; in a fresh clone, place datasets under the same
`data/` paths or override them from the CLI:

```bash
hitl-run --config configs/gpqa_baseline_llama8b.example.yaml --input data/external/gpqa_diamond_mc_shuffled --dry-run
```

## Strategies

| Strategy | Dataset | Reward Source | Notes |
| --- | --- | --- | --- |
| `baseline` | `gsm8k`, `gpqa` | none | Plain generation baseline using the dataset adapter. |
| `no_reflection` | `gsm8k` | PRM + OpenAI feedback | Corrects low-reward steps without self-reflection. |
| `reflection` | `gsm8k` | PRM + OpenAI feedback | Runs model self-reflection before tutor feedback. |
| `global_feedback` | `gsm8k` | PRM + OpenAI feedback | Maintains persistent global advice. |
| `self_consistency` | `gsm8k` | PRM + OpenAI feedback | Samples multiple paths and syncs after HITL correction. |
| `gpqa_hitl` | `gpqa` | OpenAI trajectory reward | Scores the whole GPQA trajectory with GPT, not PRM. |

GSM8K HITL strategies use `reward_model` as a local PRM, typically:

```yaml
reward_model: Qwen/Qwen2.5-Math-PRM-7B
```

`gpqa_hitl` uses `reward_model` as the OpenAI trajectory grader and
`openai_model` as the tutor feedback model:

```yaml
reward_model: gpt-5-mini
openai_model: gpt-5
```

## Example Runs

GSM8K no-reflection:

```bash
hitl-run --config configs/gsm8k_no_reflection_llama8b.example.yaml --limit 3
```

GSM8K reflection:

```bash
hitl-run --config configs/gsm8k_reflection_llama8b.example.yaml --limit 3
```

GSM8K global feedback:

```bash
hitl-run --config configs/gsm8k_global_feedback_hf_llama8b.example.yaml --limit 3
```

Choose a global feedback legacy variant with:

```yaml
global_feedback_mode: hf          # hf | hf_no_reason | hc | sf
```

GSM8K self-consistency:

```bash
hitl-run --config configs/gsm8k_self_consistency_llama8b.example.yaml --limit 3
```

GPQA baseline:

```bash
hitl-run --config configs/gpqa_baseline_llama8b.example.yaml --limit 3
```

GPQA HITL with API trajectory reward:

```bash
hitl-run --config configs/gpqa_hitl_api_llama8b.example.yaml --limit 3
```

## Config Overrides

Any config value can be overridden from the CLI. For example:

```bash
hitl-run \
  --config configs/gsm8k_self_consistency_llama8b.example.yaml \
  --num-samples 2 \
  --limit 1 \
  --dry-run
```

Direct CLI invocation also works:

```bash
hitl-run \
  --dataset gsm8k \
  --strategy no_reflection \
  --input data/processed/gsm8k_test_challenge_common_filtered.jsonl \
  --output outputs/runs/gsm8k_no_reflection_llama8b.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --reward-model Qwen/Qwen2.5-Math-PRM-7B \
  --max-steps 17 \
  --resume
```

## Example Configs

```text
configs/gsm8k_no_reflection_llama8b.example.yaml
configs/gsm8k_reflection_llama8b.example.yaml
configs/gsm8k_global_feedback_hf_llama8b.example.yaml
configs/gsm8k_global_feedback_hc_llama8b.example.yaml
configs/gsm8k_global_feedback_sf_llama8b.example.yaml
configs/gsm8k_self_consistency_llama8b.example.yaml
configs/gpqa_baseline_llama8b.example.yaml
configs/gpqa_hitl_api_llama8b.example.yaml
```

## Outputs And Archive

Generated outputs should go under `outputs/runs/`; this directory is ignored by
git except for `.gitkeep`.

Large datasets, model checkpoints, Hugging Face caches, virtual environments,
logs, generated artifacts, and the local `archive/` directory are intentionally
ignored.
