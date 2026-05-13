# Experiments

Use this directory for lightweight experiment planning, notebooks, and analysis
notes that are useful to keep in git.

Recommended contents:

- `notebooks/`: exploratory notebooks or small analysis notebooks.
- `runbooks/`: notes for named experiment batches.
- `tables/`: small hand-written summaries or markdown result tables.

Keep reusable execution settings in `configs/`, not here. Generated run outputs
should go to `outputs/runs/`.

Example workflow:

```bash
hitl-run --config configs/gsm8k_reflection_llama8b.example.yaml --limit 100
```
