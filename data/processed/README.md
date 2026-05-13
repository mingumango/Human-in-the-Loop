# Processed Data

Place experiment-ready datasets here.

GSM8K JSONL rows should include at least:

```json
{
  "index": 0,
  "question": "...",
  "ground_truth_final": "42"
}
```

The runner can read these files with:

```bash
hitl-run --dataset gsm8k --input data/processed/gsm8k_test_common_filtered.jsonl
```
