# Paper Utils — JitGen Artifact Generation

This directory contains the notebook and outputs for generating publication-quality figures and tables for the JitGen research paper. The artifacts compare **async (JitGen)** — incremental, streaming code execution — against **sync (sequential)** — traditional generate-then-execute — across 5 LLMs on the MBPP benchmark.

## Quick Start

```bash
uv sync
# Open artifacts.ipynb in VS Code / Jupyter and run all cells
```

## Models Evaluated

| Model | Key |
|---|---|
| Gemma 3 | `gemma3_latest` |
| Llama 3.2 | `llama3_2_latest` |
| Phi 4 | `phi4_latest` |
| GPT-OSS 20B | `gpt-oss_20b` |
| Qwen3-Coder 30B | `qwen3-coder_30b` |

## Metrics

| Metric | Definition |
|---|---|
| **Execution Time** | Total wall-clock time from invocation start to completion (seconds). |
| **First Output Time** | Time until the first output chunk is emitted (seconds). This is the key JitGen advantage — async execution produces visible output before the LLM finishes generating. |
| **Pass Rate** | Percentage of tasks that execute without errors (`1 − ErrorOccurred`). |
| **Correctness Rate** | Percentage of tasks whose `ActualOutput` matches `ExpectedOutput`. |
| **Timeout Rate** | Percentage of tasks that exceeded the 30-second execution timeout. |
| **ΔFirstOutput** | Per-task paired difference: `AsyncFirstOutput − SyncFirstOutput`. Negative values indicate JitGen is faster. |
| **Speedup Ratio** | `SyncFirstOutput / AsyncFirstOutput`. Values > 1 indicate JitGen delivers first output faster. |
| **Wilcoxon W** | Test statistic from the Wilcoxon signed-rank test (paired, non-parametric). |
| **Rank-biserial r** | Effect size: proportion of concordant pairs minus discordant pairs. Ranges from −1 to +1. |

---

## Tables (`tables/`)

### `summary_statistics.csv`
Descriptive statistics for Execution Time and First Output Time, grouped by (Model, Algorithm).

| Column | Description |
|---|---|
| `Count` | Number of observations |
| `ExecTime_Mean`, `ExecTime_Median`, `ExecTime_Std` | Mean, median, and standard deviation of total execution time (s) |
| `FirstOutput_Mean`, `FirstOutput_Median`, `FirstOutput_Std` | Mean, median, and standard deviation of time to first output (s) |

**Paper use:** Main results table — shows that median first-output time is consistently lower for async while total execution time remains comparable.

### `pass_correctness.csv`
Pass rate, correctness rate, and timeout rate per (Model, Algorithm).

| Column | Description |
|---|---|
| `Count` | Number of tasks |
| `PassRate` | % of tasks without execution errors |
| `CorrectnessRate` | % of tasks producing correct output |
| `TimeoutRate` | % of tasks that timed out |

**Paper use:** Demonstrates that JitGen's incremental execution does not sacrifice correctness — async and sync rates are identical or near-identical for each model.

### `statistical_tests.csv`
Wilcoxon signed-rank tests with Bonferroni correction for paired (Async − Sync) deltas.

| Column | Description |
|---|---|
| `Model` | Model name |
| `Metric` | `ExecutionTime` or `FirstOutput` |
| `N` | Number of non-zero paired differences |
| `MeanDelta`, `MedianDelta` | Mean and median of the paired difference |
| `W` | Wilcoxon test statistic |
| `p_value` | Raw p-value (two-sided) |
| `p_corrected` | Bonferroni-corrected p-value (×10 tests) |
| `EffectSize_r` | Rank-biserial correlation |
| `Significant` | `Yes` if p_corrected < 0.05 |

**Paper use:** Provides statistical evidence that first-output time differences are significant (p < 0.001 for all 5 models after correction).

### `error_breakdown.csv`
Cross-tabulation of error categories by (Model, Algorithm).

| Error Category | Description |
|---|---|
| `Success` | No error |
| `Timeout` | Exceeded execution time limit |
| `Disabled input()` | LLM generated code calling `input()`, which is sandboxed |
| `TypeError`, `IndexError`, `NameError`, `SyntaxError`, `ValueError`, `AttributeError` | Python exception categories |
| `Other Error` | Unclassified errors |

**Paper use:** Shows error profiles are symmetric between async and sync — JitGen does not introduce new failure modes.

### `summary_by_error.csv` / `summary_delta.csv`
Legacy tables from earlier analysis runs. Retained for backward compatibility.

---

## Figures (`images/`)

### `correctness_barplot.png`
**Type:** Grouped bar chart  
**Axes:** x = Model, y = Correctness Rate (%), grouped by Algorithm  
**Key insight:** Async and sync bars are identical for each model, proving JitGen preserves correctness.

### `first_output_violin.png` ⭐ Key Figure
**Type:** Violin plots faceted by model  
**Axes:** x = Algorithm, y = Time to First Output (s)  
**Key insight:** Async violins show lower medians and tighter distributions — JitGen produces user-visible output significantly earlier than sequential execution.

### `execution_time_violin.png`
**Type:** Violin plots faceted by model  
**Axes:** x = Algorithm, y = Total Execution Time (s)  
**Key insight:** Total execution times are comparable between async and sync, confirming JitGen adds negligible overhead.

### `first_output_ecdf.png`
**Type:** Empirical CDF (two panels: aggregate + per-model)  
**Axes:** x = Time to First Output (s), y = Cumulative Proportion  
**Key insight:** The async CDF curve is shifted left (faster) with median annotations. More rigorous than histograms — avoids bin-width sensitivity.

### `delta_first_output_boxplot.png`
**Type:** Box + strip plot with significance annotations  
**Axes:** x = Model, y = ΔFirstOutput (Async − Sync) in seconds  
**Key insight:** Boxes sit below zero (JitGen faster) with *** annotations from Wilcoxon tests. Shows per-task consistency — not just aggregate means.

### `speedup_ratio_barplot.png`
**Type:** Bar chart with IQR error bars  
**Axes:** x = Model, y = Median Speedup Ratio (Sync / Async)  
**Key insight:** All models above 1.0× baseline; Gemma 3 achieves 1.22×, easily citable as "N× speedup."

### `error_distribution.png`
**Type:** Stacked horizontal bar chart  
**Axes:** y = Model × Algorithm, x = Percentage (%)  
**Key insight:** Error category proportions are symmetric between async and sync — JitGen does not change failure behavior.

### `summary_heatmap.png`
**Type:** Annotated heatmap (normalized per column, raw values annotated)  
**Rows:** Models, **Columns:** PassRate / Correctness / Median ExecTime / Median FirstOutput × Algorithm  
**Key insight:** Compact overview for appendix — visually highlights that MdnFirstOutput(Async) < MdnFirstOutput(Sync) in every row.

### Legacy figures
The following are retained from earlier analysis (`plots.ipynb`):

| File | Description |
|---|---|
| `execution_time_catplot.png` | Box plots of execution time faceted by model |
| `first_output_catplot.png` | Box plots of first output time faceted by model |
| `execution_time_hist.png` | Overlaid histograms of execution time (Async vs Sync) |
| `first_output_hist.png` | Overlaid histograms of first output time |
| `execution_time_boxplot.png` | Box plot grouped by (ErrorOccurred, Algorithm) |
| `delta_execution_time_boxplot.png` | Delta execution time box plot per model |

---

## Statistical Methods

- **Wilcoxon signed-rank test** (two-sided): Non-parametric paired test for comparing async vs sync times on the same tasks. Appropriate because execution times are non-normal (right-skewed).
- **Bonferroni correction**: Conservative multiple-comparison adjustment across 5 models × 2 metrics = 10 tests (α = 0.05).
- **Rank-biserial correlation (r)**: Effect size computed as `r = 1 − 2W / [n(n+1)/2]`. Interpretable as the net proportion of pairs favoring one approach.

## Dependencies

See [`pyproject.toml`](pyproject.toml): `matplotlib`, `pandas`, `seaborn`, `scipy`, `ipykernel`.
