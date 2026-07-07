# Function Detective Analysis

- Input folder analyzed: `/Users/shahrukh/Documents/Shahrukh_Thesis/Alien-Abduction/results/function_detective__instance_passive_examples`
- Output folder: `/Users/shahrukh/Documents/Shahrukh_Thesis/Alien-Abduction/analysis_runs/function_detective__instance_passive_examples_analysis`
- Models found: Qwen3.6-35B-A3B-FP8, Qwen3.6-35B-A3B-FP8-without-reasoning, gpt-4.1-2025-04-14, gpt-4.1-mini, gpt-4.1-nano
- Modes found: passive_examples
- Domains found: list, logic, numbers, string, two_numbers
- Test-case counts detected: 2, 4, 100

## Scoring

- Accuracy = 1 for an exact hidden-function recovery, else 0.
- Interactive efficiency = `(T_total - T_used + 1) / T_total`, clamped to `[0, 1]`.
- One-shot modes do not use the tolerance term and do not multiply correctness by efficiency.
- Interactive quality score = `100 * Accuracy * Efficiency`.
- One-shot quality score = `100 * Accuracy`.

## Auto Findings

- Strongest model overall: `Qwen3.6-35B-A3B-FP8`
- Strongest domain by correct guesses: `numbers`
- Weakest domain: `logic`
- Strongest callable: `abs_diff`
- Weakest callable: `uppercase_str`
- Biggest failure reason: `aborted`
- Average failed-episode observed-pair consistency: `0.000`
- Failure pattern read: `mostly inconsistent`

## Figures

- [fig_01_model_comparison_bar](figures/fig_01_model_comparison_bar.png)
- [fig_02_domain_comparison_bar](figures/fig_02_domain_comparison_bar.png)
- [fig_03_success_rate_by_domain](figures/fig_03_success_rate_by_domain.png)
- [fig_04_passed_failed_testcases_stacked](figures/fig_04_passed_failed_testcases_stacked.png)
- [fig_05_pass_rate_heatmap](figures/fig_05_pass_rate_heatmap.png)
- [fig_06_failed_zero_accuracy_consistency](figures/fig_06_failed_zero_accuracy_consistency.png)
- [fig_06b_failed_zero_accuracy_consistency_by_domain](figures/fig_06b_failed_zero_accuracy_consistency_by_domain.png)
- [fig_07_failed_zero_accuracy_consistency_distribution](figures/fig_07_failed_zero_accuracy_consistency_distribution.png)
- [fig_09_failure_reason_stacked](figures/fig_09_failure_reason_stacked.png)
- [fig_10_protocol_violation_bar](figures/fig_10_protocol_violation_bar.png)
- [fig_11_efficiency_distribution](figures/fig_11_efficiency_distribution.png)
- [fig_12_efficiency_vs_success](figures/fig_12_efficiency_vs_success.png)
- [fig_13_callable_success_dotplot](figures/fig_13_callable_success_dotplot.png)
- [fig_14_callable_consistency_dotplot](figures/fig_14_callable_consistency_dotplot.png)
- [fig_17_budget_used_by_domain_outcome](figures/fig_17_budget_used_by_domain_outcome.png)
- [fig_18_guess_turn_by_domain_outcome](figures/fig_18_guess_turn_by_domain_outcome.png)

## Tables

- [callable_summary](callable_summary.csv)
- [domain_summary](domain_summary.csv)
- [failure_reason_summary](failure_reason_summary.csv)
- [guess_budget_summary](guess_budget_summary.csv)
- [model_summary](model_summary.csv)
- [table_01_model_summary](tables/table_01_model_summary.csv)
- [table_02_domain_summary](tables/table_02_domain_summary.csv)
- [table_03_model_domain_summary](tables/table_03_model_domain_summary.csv)
- [table_04_failure_reason_summary](tables/table_04_failure_reason_summary.csv)
- [table_05_callable_summary](tables/table_05_callable_summary.csv)
- [table_06_consistency_summary](tables/table_06_consistency_summary.csv)
- [table_07_protocol_metrics](tables/table_07_protocol_metrics.csv)
- [table_08_scoring_components](tables/table_08_scoring_components.csv)
- [table_09_guess_budget_summary](tables/table_09_guess_budget_summary.csv)
