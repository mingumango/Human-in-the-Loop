# Runbooks

Use one markdown file per experiment batch. A good runbook records:

- config files used
- dataset path and split
- model and reward model
- command line overrides
- output paths
- short notes on failures or reruns

Keep this directory small and human-readable. Put generated JSONL results under
`outputs/runs/`.
