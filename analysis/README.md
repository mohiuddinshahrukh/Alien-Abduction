# Alien Abduction Analysis

This pipeline analyzes exactly one clemcore results folder at a time and writes all derived outputs outside the results tree.

Usage:

```bash
cd /Users/shahrukh/Documents/Shahrukh_Thesis/Alien-Abduction
.venv/bin/python analysis/run_all.py --input results/passive_examples --name passive_examples_analysis
```

Filter to one model inside a shared results folder:

```bash
cd /Users/shahrukh/Documents/Shahrukh_Thesis/Alien-Abduction
.venv/bin/python analysis/run_all.py --input results/passive_examples --model gpt-4-0613 --name gpt_4_0613_passive_examples_analysis
```

Output root:

```text
/Users/shahrukh/Documents/Shahrukh_Thesis/Alien-Abduction/analysis_runs/<analysis_name>/
```

Main artifacts:

- `episodes.csv`
- `model_summary.csv`
- `domain_summary.csv`
- `callable_summary.csv`
- `failure_reason_summary.csv`
- `tables/*.csv`
- `figures/*.png`
- `figures/*.pdf`
- `report.md`

Notes:

- The script reads only the folder passed in `--input`.
- If `--model` is provided, only episodes for that exact model name are analyzed.
- The script never writes into the input results folder.
- If the output analysis folder already exists, it is replaced to avoid stale figures and tables.
