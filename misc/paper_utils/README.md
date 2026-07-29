# Paper Utils — JitGen Artifact Generation

This directory contains the notebook and outputs for generating publication-quality figures and tables for the JitGen article submitted to *Technology Audit and Production Reserves* (TARP). The artifacts compare **Async (JitGen)** — incremental, streaming code execution — against **Sync** — traditional generate-then-execute — across 6 LLMs on the MBPP benchmark.

## Quick Start

```bash
uv sync
# Open artifacts.ipynb in VS Code / Jupyter and run all cells, or headless:
uv run --with jupyter,nbconvert jupyter nbconvert --to notebook --execute --inplace artifacts.ipynb
```

## Models Evaluated

| Model | Key |
|---|---|
| Gemma 3 | `gemma3_latest` |
| Llama 3.2 | `llama3_2_latest` |
| Phi 4 | `phi4_latest` |
| GPT-OSS 20B | `gpt-oss_20b` |
| Qwen3-Coder 30B | `qwen3-coder_30b` |
| CodeAct-Mistral | `xingyaow__codeact-agent-mistral` |

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
| `p_corrected` | Bonferroni-corrected p-value (×12 tests) |
| `EffectSize_r` | Rank-biserial correlation |
| `Significant` | `Yes` if p_corrected < 0.05 |

**Paper use:** Provides statistical evidence that first-output time differences are significant (p < 0.001 for all 6 models after correction).

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

All figures follow the TARP journal format: **170 mm printed width at 300 DPI** (≈2008 px, insert into Word at 100 % scale), all in-figure text **14 pt Times New Roman**, no in-figure titles (captions live in the manuscript), every color/hatch/linestyle decoded by an in-figure legend, both axes labeled with units on every panel, and the Async/Sync pair distinguishable in grayscale (hatch + linestyle, not color alone). Figure numbers match the TARP manuscript (`draft_tarp.md`); Figs 1–2 of the article are pseudocode and not generated here. The final notebook cell checks pixel width, height, and file size for every figure.

### `fig-first-output-violin.png` — Fig. 3 ⭐ Main figure
**Type:** Grouped violins (Async vs Sync per model), log-scale Y  
**Axes:** x = Model, y = Time to first output, s (log scale)  
**Key insight:** Async violins show lower medians — JitGen produces user-visible output earlier than sequential execution.

### `fig-first-output-ecdf.png` — Fig. 4
**Type:** ECDF, 3×2 grid, one model per panel (*a*–*f*), Async solid / Sync dashed  
**Axes:** x = Time to first output, s (log scale), y = Fraction of tasks  
**Key insight:** The async CDF curve is shifted left (faster). More rigorous than histograms — avoids bin-width sensitivity.

### `fig-delta-first-output-boxplot.png` — Fig. 5
**Type:** Box plot of paired per-task differences with Δ = 0 reference line  
**Axes:** x = Model, y = Δ = Async − Sync, s (axis clipped; CodeAct-Mistral median annotated)  
**Key insight:** Boxes sit below zero (JitGen faster). Shows per-task consistency — not just aggregate means. Significance lives in `statistical_tests.csv`.

### `fig-execution-time-violin.png` — Fig. 6
**Type:** Grouped violins, same layout and log-Y decision as Fig. 3  
**Axes:** x = Model, y = Execution time, s (log scale)  
**Key insight:** Total execution times are comparable between async and sync, confirming JitGen adds negligible overhead.

### `fig-speedup-ratio-barplot.png` — Fig. 7
**Type:** Bar chart with IQR error bars, values printed above bars, 1.0× reference line  
**Axes:** x = Model, y = Median speedup, × (Sync / Async)  
**Key insight:** All models above 1.0× baseline; CodeAct-Mistral achieves 4.58×.

### `fig-correctness-barplot.png` — Fig. 8
**Type:** Grouped bars, 4 series (Pass/Correctness × Async/Sync); metric by color, strategy by hatch; values on bars  
**Axes:** x = Model, y = Rate, % (fixed 0–100)  
**Key insight:** Async and sync bars are identical (or near-identical) for each model, proving JitGen preserves correctness.

### `fig-error-distribution.png` — Fig. 9
**Type:** Grouped bars of outcome counts, 3×2 grid, one model per panel (*a*–*f*), symlog Y so zero counts stay visible, counts annotated  
**Axes:** x = Outcome category, y = Number of tasks (log scale)  
**Key insight:** Error category counts are symmetric between async and sync — JitGen does not change failure behavior.

### `fig-summary-heatmap.png` — Fig. 10
**Type:** Annotated heatmap (color per-column min–max normalized, raw values annotated, horizontal colorbar)  
**Rows:** Model + strategy (12), **Columns:** Pass rate, % / Correctness, % / Median exec. time, s / Median first output, s  
**Key insight:** Compact overview — median first output (Async) < (Sync) in every model pair.

---

## Statistical Methods

- **Wilcoxon signed-rank test** (two-sided): Non-parametric paired test for comparing async vs sync times on the same tasks. Appropriate because execution times are non-normal (right-skewed).
- **Bonferroni correction**: Conservative multiple-comparison adjustment across 6 models × 2 metrics = 12 tests (α = 0.05).
- **Rank-biserial correlation (r)**: Effect size computed as `r = 1 − 2W / [n(n+1)/2]`. Interpretable as the net proportion of pairs favoring one approach.

## Dependencies

See [`pyproject.toml`](pyproject.toml): `matplotlib`, `pandas`, `seaborn`, `scipy`, `ipykernel`.
