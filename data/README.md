# Data

Keep local datasets here. Large datasets are ignored by git; commit only small
metadata files or documentation that helps someone recreate the layout.

Recommended layout:

```text
data/raw/          Original downloaded files, unchanged
data/processed/    Filtered or normalized JSONL used by experiments
data/external/     Third-party dataset exports, such as saved Hugging Face data
```

Typical paths used by configs:

```text
data/processed/gsm8k_test_common_filtered.jsonl
data/external/gpqa_diamond_mc_shuffled/
```

Fresh clones should place datasets in these paths or override `--input` from
the command line.
