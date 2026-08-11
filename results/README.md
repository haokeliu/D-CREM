# Result directory

Experiment entry points write Protocol-v2 JSON records below this directory.
The initial source release contains no run outputs. Formal per-run JSON files
should be versioned with their experiment batch; large analysis caches and
generated figures remain ignored.

After an experiment batch, refresh the structured summary with:

```bash
python scripts/build_results_report.py
```

Generate a human-readable Markdown report only when needed:

```bash
python scripts/build_results_report.py --markdown results/tables/results_report.md
```
