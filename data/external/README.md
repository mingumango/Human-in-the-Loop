# External Data

Place third-party exported datasets here, especially when the original format is
already useful to the adapter.

For GPQA Diamond, the adapter supports JSONL or a Hugging Face dataset saved to
disk:

```text
data/external/gpqa_diamond_mc_shuffled/
```

Example:

```bash
hitl-run --dataset gpqa --input data/external/gpqa_diamond_mc_shuffled
```
