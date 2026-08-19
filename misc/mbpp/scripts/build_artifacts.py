from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.text import Text
from PIL import Image
from scipy.stats import wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
IMAGES_DIR = RESULTS_DIR / "images"
TABLES_DIR = RESULTS_DIR / "tables"

PAIRED_METRICS = ("ExecutionTime", "FirstExecutedStatement")
MIN_NONZERO_N = 10
ALPHA = 0.05

_METRIC_LABELS = {
    "ExecutionTime": "Execution Time",
    "FirstExecutedStatement": "First Statement",
}

FONT_SIZE_PT = 14
FIGURE_DPI = 300
MM_PER_IN = 25.4
PAGE_WIDTH_MM = 170.0  # standard figure width; every figure shares it

STRATEGIES = ["incremental", "sequential"]
PALETTE = {"incremental": "tab:blue", "sequential": "tab:orange"}


def _round_or_na(value: float, decimals: int) -> object:
    """Round one value, or return `"N/A"` for `NaN` so no cell is left empty.

    Args:
        value: The value to round.
        decimals: Number of decimal places to round to.

    Returns:
        `"N/A"` if `value` is `NaN`, else `round(value, decimals)`.
    """
    return "N/A" if pd.isna(value) else round(value, decimals)


def _format_int_or_na(value: float) -> object:
    """Render a token count as a plain integer, or `"N/A"` for `NaN`.

    Args:
        value: A token-count sum, possibly `NaN` if the run predates the
            usage column it's read from.

    Returns:
        `"N/A"` if `value` is `NaN`, else the value rounded to the nearest int.
    """
    return "N/A" if pd.isna(value) else round(value)


def _format_p(value: float) -> str:
    """Format a p-value for a results table without rounding it to a meaningless 0.

    Args:
        value: A p-value, or `NaN` for a row that wasn't tested.

    Returns:
        `"N/A"` for `NaN`; scientific notation for values below 0.001, since
        rounding those to a fixed number of decimals would just print
        `"0.000"`; otherwise a plain 3-decimal value.
    """
    if pd.isna(value):
        return "N/A"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _configure_matplotlib() -> None:
    """Set up fonts/sizes/background so every figure meets the same print spec.

    Called at the top of every `build_*_figure()` -- cheap and idempotent,
    mirroring how every `build_*_table()` independently calls
    `tables_dir.mkdir(parents=True, exist_ok=True)`. `fallback_to_default=False`
    turns a missing Times New Roman into a hard failure here rather than a
    silent fall-back to DejaVu Sans that only gets caught by eye later.

    Never call `seaborn.set_theme()`/`plt.style.use()` anywhere in this
    module -- either would silently override the rcParams set here.
    """
    font_manager.findfont(
        font_manager.FontProperties(family="Times New Roman"),
        fallback_to_default=False,
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.hspace": 0.12,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _flatten_to_opaque_rgb(path: Path, dpi: float = FIGURE_DPI) -> None:
    """Composite a saved PNG onto opaque white and drop its alpha channel.

    matplotlib's Agg backend writes an RGBA PNG even with `facecolor="white"`
    (fully opaque, but still technically an alpha channel), so this is a
    separate, explicit step rather than relying on `savefig` alone.

    Args:
        path: PNG file to flatten in place.
        dpi: Resolution to stamp back onto the file -- Pillow drops DPI
            metadata on save unless it's passed explicitly.
    """
    # PNG's pHYs chunk stores pixels-per-meter as an integer, so a requested
    # dpi of exactly 300 round-trips (on read-back) to ~299.9994, not 300 --
    # a small margin keeps `img.info["dpi"] >= (300, 300)` checks honest
    # without changing the actual rendered pixel count (fixed by `figsize`
    # x `FIGURE_DPI` at `fig.savefig()` time).
    stamped_dpi = dpi + 0.5
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        flattened = Image.alpha_composite(background, rgba).convert("RGB")
        flattened.save(path, dpi=(stamped_dpi, stamped_dpi))


def new_fig(height_mm: float) -> tuple[Figure, Axes]:
    """Create a single-axes figure at the shared standard width.

    The standard sizing helper: every figure shares `PAGE_WIDTH_MM`
    (170mm) so only height varies per figure.

    Args:
        height_mm: Figure height in millimeters.

    Returns:
        A `(fig, ax)` pair from `plt.subplots()`, styled via
        `_configure_matplotlib()`.
    """
    _configure_matplotlib()
    width_in = PAGE_WIDTH_MM / MM_PER_IN
    height_in = height_mm / MM_PER_IN
    return plt.subplots(figsize=(width_in, height_in))


def save_fig(fig: Figure, filename: str, images_dir: Path = IMAGES_DIR) -> None:
    """Enforce the figure spec and write one flattened, opaque PNG.

    The single point every `build_*_figure()` calls instead of repeating
    savefig/flatten logic. `bbox_inches` is deliberately left at its default
    (i.e. not `"tight"`): a post-save bbox crop would change the saved canvas
    size and break the `pixels / dpi == PAGE_WIDTH_MM` identity the whole
    physical-size/14pt contract depends on -- whitespace trimming instead
    comes from `constrained_layout` (set in `_configure_matplotlib()`) at
    layout time. Whoever places the PNG in the manuscript must insert it at
    natural size (no `width=` rescaling), or that contract is lost outside
    this script's control.

    Args:
        fig: The figure to save. Closed after saving.
        filename: PNG filename (not a full path), e.g. `"fig-foo.png"`.
        images_dir: Directory the PNG is written into.
    """
    for text in fig.findobj(Text):
        if text.get_text().strip():
            assert text.get_fontsize() == FONT_SIZE_PT, text.get_text()
    for ax in fig.axes:
        assert ax.get_xlabel().strip(), ax
        assert ax.get_ylabel().strip(), ax

    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / filename
    fig.savefig(path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)
    _flatten_to_opaque_rgb(path)


def load_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Load every result CSV in `results_dir` into one combined `DataFrame`.

    Args:
        results_dir: Directory containing one CSV per (model, strategy) pass,
            as written by `mbpp.results.write_csv`.

    Returns:
        The concatenated rows of every CSV found directly under `results_dir`.

    Raises:
        FileNotFoundError: If `results_dir` contains no CSV files.
    """
    csv_paths = sorted(results_dir.glob("*.csv"))
    if not csv_paths:
        msg = f"No result CSVs found in {results_dir}"
        raise FileNotFoundError(msg)
    return pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True)


def build_timing_summary(
    df: pd.DataFrame, tables_dir: Path = TABLES_DIR
) -> pd.DataFrame:
    """Summarize execution time and time-to-first-statement by model x strategy.

    Args:
        df: Combined result rows, as returned by `load_results()`.
        tables_dir: Directory the summary CSV is written into.

    Returns:
        One row per (`ModelLabel`, `Strategy`) with count/mean/median/std for
        `ExecutionTime` and `FirstExecutedStatement`. `FirstExecutedStatement`
        is blank for rows with no executed statement (e.g. an immediate
        timeout), so its count can be lower than `ExecutionTime`'s.
    """
    summary = df.groupby(["ModelLabel", "Strategy"], sort=False).agg(
        execution_time_count=("ExecutionTime", "count"),
        execution_time_mean=("ExecutionTime", "mean"),
        execution_time_median=("ExecutionTime", "median"),
        execution_time_std=("ExecutionTime", "std"),
        first_statement_count=("FirstExecutedStatement", "count"),
        first_statement_mean=("FirstExecutedStatement", "mean"),
        first_statement_median=("FirstExecutedStatement", "median"),
        first_statement_std=("FirstExecutedStatement", "std"),
    )
    summary = summary.reset_index()

    round_columns = [
        "execution_time_mean",
        "execution_time_median",
        "execution_time_std",
        "first_statement_mean",
        "first_statement_median",
        "first_statement_std",
    ]
    for column in round_columns:
        summary[column] = summary[column].map(lambda v: _round_or_na(v, 2))

    summary["Strategy"] = summary["Strategy"].str.capitalize()
    summary = summary.rename(
        columns={
            "ModelLabel": "Model",
            "execution_time_count": "Execution Time N",
            "execution_time_mean": "Execution Time Mean (s)",
            "execution_time_median": "Execution Time Median (s)",
            "execution_time_std": "Execution Time Std (s)",
            "first_statement_count": "First Statement N",
            "first_statement_mean": "First Statement Mean (s)",
            "first_statement_median": "First Statement Median (s)",
            "first_statement_std": "First Statement Std (s)",
        }
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "timing_summary.csv", index=False)
    return summary


def _no_code_generated(frame: pd.DataFrame | pd.Series) -> pd.Series | bool:
    """True where an incremental run finished `ok` without ever executing a statement.

    Mirrors sequential's harness-level `GenerationError` (`_extract_code_block`
    finding no ` ```python ` fence anywhere in the response, `execution.py`):
    incremental has no equivalent exception for that case -- a response with
    no code just yields zero dispatched statements, which reads as an empty,
    otherwise-unremarkable `ok` run unless specifically checked for here.
    `FirstExecutedStatement` being unset is the unambiguous signal, not blank
    `ExecutionOutput`: a block that genuinely executes but never prints
    anything also has blank `ExecutionOutput`, but *does* have a
    `FirstExecutedStatement` timestamp, since the executor still ran
    something. Used both to fold these rows into "failed" for the pass rate
    and to classify them as `"Generation Error"` in the breakdown table, so
    they line up with sequential's identical-concept failures instead of
    silently inflating incremental's numbers.

    Args:
        frame: Either the full combined `DataFrame` (vectorized use) or a
            single row `Series` (from `df.apply(..., axis=1)`) -- `pd.isna`
            handles both shapes uniformly, unlike the `.isna()` method, which
            only exists on the `DataFrame`/`Series` shape, not on the bare
            scalar a row's `FirstExecutedStatement` lookup produces.

    Returns:
        A boolean `Series` for a `DataFrame` input, or a plain `bool` for a
        row `Series` input.
    """
    return (
        (frame["Strategy"] == "incremental")
        & (frame["Outcome"] == "ok")
        & pd.isna(frame["FirstExecutedStatement"])
    )


def build_pass_correctness(
    df: pd.DataFrame, tables_dir: Path = TABLES_DIR
) -> pd.DataFrame:
    """Compute pass/correctness/timeout rates by model x strategy.

    "Pass" (didn't error, didn't time out, and -- for incremental -- actually
    ran some generated code) and "correct" (produced the right output) are
    kept as separate rates -- a run can execute cleanly and still be wrong.
    A timeout counts as a failure here even though `ErrorOccurred` alone
    wouldn't flag it (`results.py`'s `ErrorOccurred` is `status == "error"`
    specifically, with `HasTimedOut` a separate column for `status ==
    "timeout"`) -- a stalled run is not a pass. Incremental's silent
    "no code, zero statements executed" success (see `_no_code_generated`)
    is folded in the same way, so it counts as a failure exactly like
    sequential's `GenerationError` for the identical underlying response.

    Args:
        df: Combined result rows, as returned by `load_results()`.
        tables_dir: Directory the rates CSV is written into.

    Returns:
        One row per (`ModelLabel`, `Strategy`) with the run count and the
        pass/correctness/timeout rates, as percentages.
    """
    working = df.assign(
        Failed=df["ErrorOccurred"] | df["HasTimedOut"] | _no_code_generated(df)
    )
    summary = working.groupby(["ModelLabel", "Strategy"], sort=False).agg(
        Count=("Failed", "size"),
        PassRate=("Failed", lambda s: (1 - s.mean()) * 100),
        CorrectnessRate=("CorrectOutput", lambda s: s.mean() * 100),
        TimeoutRate=("HasTimedOut", lambda s: s.mean() * 100),
    )
    summary = summary.reset_index()

    rate_columns = ["PassRate", "CorrectnessRate", "TimeoutRate"]
    summary[rate_columns] = summary[rate_columns].round(2)

    summary["Strategy"] = summary["Strategy"].str.capitalize()
    summary = summary.rename(
        columns={
            "ModelLabel": "Model",
            "Count": "N",
            "PassRate": "Pass Rate (%)",
            "CorrectnessRate": "Correctness Rate (%)",
            "TimeoutRate": "Timeout Rate (%)",
        }
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "pass_correctness.csv", index=False)
    return summary


def _paired_wide(df: pd.DataFrame, model_label: str, metric: str) -> pd.DataFrame:
    """Pivot incremental vs. sequential per-task values for one model and metric.

    Args:
        df: Combined result rows, as returned by `load_results()`.
        model_label: The `ModelLabel` to restrict to.
        metric: Column to pair, e.g. `"ExecutionTime"`.

    Returns:
        `incremental`/`sequential` columns indexed by `DatasetRow`, for tasks
        where both strategies have a value for `metric` -- pairing on
        `DatasetRow` is valid since each dataset row maps to exactly one
        case per strategy.
    """
    subset = df.loc[df["ModelLabel"] == model_label, ["DatasetRow", "Strategy", metric]]
    wide = subset.pivot_table(
        index="DatasetRow", columns="Strategy", values=metric, aggfunc="first"
    )
    wide = wide.reindex(columns=["incremental", "sequential"])
    return wide.dropna(subset=["incremental", "sequential"])


def _paired_deltas(df: pd.DataFrame, model_label: str, metric: str) -> pd.Series:
    """Pair incremental vs. sequential per-task values for one model and metric.

    Args:
        df: Combined result rows, as returned by `load_results()`.
        model_label: The `ModelLabel` to restrict to.
        metric: Column to pair, e.g. `"ExecutionTime"`.

    Returns:
        `incremental - sequential` per `DatasetRow` -- an absolute delta,
        which is what the paired significance test in `_compute_significance`
        needs (it's testing raw time differences, not ratios).
    """
    wide = _paired_wide(df, model_label, metric)
    return wide["incremental"] - wide["sequential"]


def _compute_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Test whether incremental vs. sequential deltas are significant, by model x metric.

    For each model and each of `PAIRED_METRICS`, the per-task delta
    (incremental - sequential) is tested with a two-sided Wilcoxon
    signed-rank test. Exact zero-deltas are dropped first, since Wilcoxon
    can't handle ties at zero; if fewer than `MIN_NONZERO_N` non-zero paired
    observations remain, the row is emitted with the test statistic/p-value
    left as `NaN` rather than run through `scipy`. Because one test is run
    per (model, metric) combination, p-values are Bonferroni-corrected
    across every test actually run.

    Kept separate from `build_significance_table` so other consumers of these
    raw Wilcoxon results don't have to parse a display-formatted table back
    into numbers.

    Args:
        df: Combined result rows, as returned by `load_results()`.

    Returns:
        One row per (model, metric) with the paired deltas, the Wilcoxon
        statistic/p-value, the Bonferroni-corrected p-value, the
        rank-biserial effect size, and a `Significant` verdict at `ALPHA` --
        all unrounded, with `Metric` still holding the raw column name.
    """
    rows: list[dict[str, object]] = []
    for model_label in df["ModelLabel"].unique():
        for metric in PAIRED_METRICS:
            deltas = _paired_deltas(df, model_label, metric)
            nonzero = deltas[deltas != 0]
            n = len(nonzero)

            row: dict[str, object] = {
                "Model": model_label,
                "Metric": metric,
                "N": n,
                "MeanDelta": deltas.mean(),
                "MedianDelta": deltas.median(),
            }
            if n < MIN_NONZERO_N:
                row.update(
                    W=float("nan"),
                    p_value=float("nan"),
                    EffectSize_r=float("nan"),
                    Significant="N/A (n<10)",
                )
            else:
                statistic, p_value = wilcoxon(nonzero, alternative="two-sided")
                row.update(
                    W=statistic,
                    p_value=p_value,
                    EffectSize_r=1 - (2 * statistic) / (n * (n + 1) / 2),
                    Significant=None,
                )
            rows.append(row)

    table = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Metric",
            "N",
            "MeanDelta",
            "MedianDelta",
            "W",
            "p_value",
            "EffectSize_r",
            "Significant",
        ],
    )

    n_tests = table["p_value"].notna().sum()
    table["p_corrected"] = (table["p_value"] * n_tests).clip(upper=1.0)
    tested = table["p_value"].notna()
    table.loc[tested, "Significant"] = table.loc[tested, "p_corrected"].map(
        lambda p: "Yes" if p <= ALPHA else "No"
    )
    return table[
        [
            "Model",
            "Metric",
            "N",
            "MeanDelta",
            "MedianDelta",
            "W",
            "p_value",
            "p_corrected",
            "EffectSize_r",
            "Significant",
        ]
    ]


def build_significance_table(
    df: pd.DataFrame, tables_dir: Path = TABLES_DIR
) -> pd.DataFrame:
    """Format `_compute_significance()`'s output into the publication-ready CSV table.

    Args:
        df: Combined result rows, as returned by `load_results()`.
        tables_dir: Directory the significance-table CSV is written into.

    Returns:
        One row per (model, metric), rounded and with publication-ready headers.
    """
    table = _compute_significance(df)
    table["Metric"] = table["Metric"].map(_METRIC_LABELS)
    table["MeanDelta"] = table["MeanDelta"].map(lambda v: _round_or_na(v, 4))
    table["MedianDelta"] = table["MedianDelta"].map(lambda v: _round_or_na(v, 4))
    table["W"] = table["W"].map(lambda v: _round_or_na(v, 1))
    table["EffectSize_r"] = table["EffectSize_r"].map(lambda v: _round_or_na(v, 2))
    table["p_value"] = table["p_value"].map(_format_p)
    table["p_corrected"] = table["p_corrected"].map(_format_p)
    table = table.rename(
        columns={
            "MeanDelta": "Mean Delta (s)",
            "MedianDelta": "Median Delta (s)",
            "W": "Wilcoxon W",
            "p_value": "p-value",
            "p_corrected": "p-value (Bonferroni)",
            "EffectSize_r": "Effect Size (r)",
            "Significant": "Significant (alpha=0.05)",
        }
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "significance_tests.csv", index=False)
    return table


def _paired_token_chunks(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    """Pair per-task `ClientCompletionChunks` and the incremental arm's `EarlyExit` flag.

    `ClientCompletionChunks` (not `OutputTokens`) is what token-savings has to
    be computed from: the provider's final usage chunk never arrives when the
    stream is aborted mid-block, so every early-exited row has `OutputTokens`
    `NaN` -- exactly the rows this comparison needs.

    Args:
        df: Combined result rows, as returned by `load_results()`.
        model_label: The `ModelLabel` to restrict to.

    Returns:
        One row per `DatasetRow` with `incremental_chunks`,
        `sequential_chunks`, and `early_exit`, for tasks where both
        strategies ran.
    """
    subset = df.loc[df["ModelLabel"] == model_label]
    chunks = subset.pivot_table(
        index="DatasetRow",
        columns="Strategy",
        values="ClientCompletionChunks",
        aggfunc="first",
    ).reindex(columns=["incremental", "sequential"])
    chunks = chunks.dropna(subset=["incremental", "sequential"])
    chunks.columns = ["incremental_chunks", "sequential_chunks"]

    early_exit = subset.loc[subset["Strategy"] == "incremental"].set_index(
        "DatasetRow"
    )["EarlyExit"]
    chunks["early_exit"] = chunks.index.map(early_exit).fillna(False)
    return chunks


def _savings_pct(paired: pd.DataFrame) -> float:
    """Compute the % of output tokens (by `ClientCompletionChunks`) incremental saved.

    Args:
        paired: Rows shaped like `_paired_token_chunks()`'s output, or a
            subset of them (e.g. restricted to early-exited tasks).

    Returns:
        `(sequential_total - incremental_total) / sequential_total * 100`, or
        `NaN` if `sequential_total` is 0 -- nothing to compare savings against.
    """
    sequential_total = paired["sequential_chunks"].sum()
    if sequential_total == 0:
        return float("nan")
    incremental_total = paired["incremental_chunks"].sum()
    return (sequential_total - incremental_total) / sequential_total * 100


def build_token_usage(
    df: pd.DataFrame, tables_dir: Path = TABLES_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize token usage by model x strategy, plus incremental's abort-driven savings.

    Kept separate from `build_timing_summary` so that table's shape doesn't
    have to accommodate token columns. `ReasoningTokens_Sum` uses
    `min_count=1` so an all-`NaN` group (predates the column) stays `NaN`
    rather than collapsing to 0, which would be indistinguishable from a
    model that demonstrably used no extended thinking.

    The sibling savings table reports the % of output tokens (by
    `ClientCompletionChunks`, not `OutputTokens` -- see `_paired_token_chunks`)
    the incremental arm saved by aborting early, both overall and restricted
    to just the runs that actually aborted, since the whole-set number is
    diluted by runs that had no tail to skip.

    Args:
        df: Combined result rows, as returned by `load_results()`.
        tables_dir: Directory the usage/savings CSVs are written into.

    Returns:
        A `(token_usage, token_saving)` pair of `DataFrame`s: one row per
        (`ModelLabel`, `Strategy`) for `token_usage`, one row per
        `ModelLabel` for `token_saving`.
    """
    usage = df.groupby(["ModelLabel", "Strategy"], sort=False).agg(
        Count=("ErrorOccurred", "size"),
        EarlyExits=("EarlyExit", "sum"),
        InputTokens_Mean=("InputTokens", "mean"),
        InputTokens_Sum=("InputTokens", "sum"),
        OutputTokens_Mean=("OutputTokens", "mean"),
        OutputTokens_Sum=("OutputTokens", "sum"),
        TotalTokens_Mean=("TotalTokens", "mean"),
        TotalTokens_Sum=("TotalTokens", "sum"),
        ReasoningTokens_Mean=("ReasoningTokens", "mean"),
        ReasoningTokens_Sum=("ReasoningTokens", lambda s: s.sum(min_count=1)),
    )
    usage = usage.reset_index()

    for column in ["InputTokens_Mean", "OutputTokens_Mean", "TotalTokens_Mean"]:
        usage[column] = usage[column].round(2)
    for column in ["InputTokens_Sum", "OutputTokens_Sum", "TotalTokens_Sum"]:
        usage[column] = usage[column].map(_format_int_or_na)
    usage["ReasoningTokens_Mean"] = usage["ReasoningTokens_Mean"].map(
        lambda v: _round_or_na(v, 2)
    )
    usage["ReasoningTokens_Sum"] = usage["ReasoningTokens_Sum"].map(_format_int_or_na)

    usage["Strategy"] = usage["Strategy"].str.capitalize()
    usage = usage.rename(
        columns={
            "ModelLabel": "Model",
            "Count": "N",
            "EarlyExits": "Early Exits",
            "InputTokens_Mean": "Input Tokens Mean",
            "InputTokens_Sum": "Input Tokens Sum",
            "OutputTokens_Mean": "Output Tokens Mean",
            "OutputTokens_Sum": "Output Tokens Sum",
            "TotalTokens_Mean": "Total Tokens Mean",
            "TotalTokens_Sum": "Total Tokens Sum",
            "ReasoningTokens_Mean": "Reasoning Tokens Mean",
            "ReasoningTokens_Sum": "Reasoning Tokens Sum",
        }
    )

    savings_rows: list[dict[str, object]] = []
    for model_label in df["ModelLabel"].unique():
        paired = _paired_token_chunks(df, model_label)
        aborted = paired[paired["early_exit"]]
        savings_rows.append(
            {
                "ModelLabel": model_label,
                "PairedRuns": len(paired),
                "AbortedRuns": len(aborted),
                "OverallSavingsPct": _savings_pct(paired),
                "AbortedSavingsPct": _savings_pct(aborted)
                if len(aborted)
                else float("nan"),
            }
        )
    saving = pd.DataFrame(savings_rows)
    saving["OverallSavingsPct"] = saving["OverallSavingsPct"].map(
        lambda v: _round_or_na(v, 2)
    )
    saving["AbortedSavingsPct"] = saving["AbortedSavingsPct"].map(
        lambda v: _round_or_na(v, 2)
    )
    saving = saving.rename(
        columns={
            "ModelLabel": "Model",
            "PairedRuns": "Paired Runs",
            "AbortedRuns": "Aborted Runs",
            "OverallSavingsPct": "Overall Savings (%)",
            "AbortedSavingsPct": "Aborted-Run Savings (%)",
        }
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    usage.to_csv(tables_dir / "token_usage.csv", index=False)
    saving.to_csv(tables_dir / "token_saving.csv", index=False)
    return usage, saving


# Keyed off `mbpp.results._OUTCOME_LABELS`'s actual vocabulary
# (`{"ok": "ok", "timeout": "stalled", "error": "error"}`) -- not the
# harness-internal status names -- since that's what's literally written to
# the CSV's `Outcome` column.
_HARNESS_OUTCOME_CATEGORIES = {"ok": "Success", "stalled": "Timeout"}
_ERROR_CATEGORIES = ["Timeout", "Generation Error", "SyntaxError", "RuntimeError"]

# Both sides classify "the model generated syntactically invalid Python" the
# same way conceptually, but under different `ErrorType` strings depending on
# which layer caught it: incremental's grammar-level rejection surfaces as
# `jitgen.errors.ExtractionError` (`mbpp.main._run_case`'s `except
# JitGenError` branch falls back to `type(exc).__name__` since an
# `ExtractionError`'s message has no `"ClassName: "` prefix to parse), while a
# syntax failure caught later -- by `compile`/`exec` inside
# `InProcPythonExecutor`, which happens for either strategy whenever code is
# grammar-valid-per-Lark but rejected by CPython (e.g. `return` outside a
# function) -- is a real `SyntaxError`, reported with that prefix and so
# already `ErrorType == "SyntaxError"` verbatim. Both are aliased here to the
# same `"SyntaxError"` category so one column covers both instead of
# splitting an identical failure mode across strategies -- `"SyntaxError"`
# needs its own identity entry despite already being spelled correctly,
# since `_classify_error`'s `.get(error_type, "RuntimeError")` would
# otherwise treat an unrecognized key (including its own eventual output
# value) as a miss and collapse it into `"RuntimeError"`. `GenerationError`
# is sequential's `ErrorType` for the harness-level "nothing to run" failure
# (`mbpp.main._run_case`'s `except ValueError` branch, for
# `_extract_code_block` finding no fenced code at all) -- not a model-code
# exception, so it's routed to the fixed `"Generation Error"` category rather
# than sitting in its own exception-shaped column. Incremental has no
# exception for the identical case (a response with no opening fence just
# yields zero dispatched statements, reported as an otherwise-unremarkable
# `ok` run) -- `_classify_error` special-cases that via `_no_code_generated`
# before falling through to `ErrorType` at all, so both strategies land in
# `"Generation Error"` for the same underlying "model never produced code"
# response. Every other exception class name (`AttributeError`, `KeyError`,
# `ValueError`, ...) is collapsed into one `"RuntimeError"` bucket rather
# than kept as its own column -- the breakdown table only distinguishes
# error *phase* (compile-time vs. run-time vs. harness-level), not exact
# exception type.
_ERROR_TYPE_ALIASES = {
    "ExtractionError": "SyntaxError",
    "SyntaxError": "SyntaxError",
    "GenerationError": "Generation Error",
}


def _classify_error(row: pd.Series) -> str:
    """Map one run's `Outcome`/`ErrorType` to a single error category.

    Args:
        row: One row of the combined results, with `Outcome`, `ErrorType`,
            `Strategy`, and `FirstExecutedStatement` columns.

    Returns:
        `"Generation Error"` for incremental's silent no-code-generated `ok`
        run (`_no_code_generated`), checked first since it would otherwise be
        misread as a plain `"Success"`; `"Success"`/`"Timeout"` for a
        matching harness-level `Outcome`; `"Generation Error"`/`"SyntaxError"`
        for their respective `ErrorType`s (aliased via
        `_ERROR_TYPE_ALIASES`); otherwise `"RuntimeError"`, collapsing every
        other exception class -- and any run with a missing `ErrorType` --
        into one runtime-failure bucket.
    """
    if _no_code_generated(row):
        return "Generation Error"
    category = _HARNESS_OUTCOME_CATEGORIES.get(row["Outcome"])
    if category is not None:
        return category
    error_type = row["ErrorType"]
    if not isinstance(error_type, str):
        return "RuntimeError"
    return _ERROR_TYPE_ALIASES.get(error_type, "RuntimeError")


def build_error_breakdown(
    df: pd.DataFrame, tables_dir: Path = TABLES_DIR
) -> pd.DataFrame:
    """Break run outcomes down into error categories, by model x strategy.

    Replaces substring-matching exception messages (e.g. `"not iterable" in
    err`): that logic was reconstructing downstream what the harness's typed
    `Outcome`/`ErrorType` fields already know at the source.

    Args:
        df: Combined result rows, as returned by `load_results()`.
        tables_dir: Directory the breakdown CSV is written into.

    Note: `Generation Error` means "the model never produced any code to
    run" for both strategies, but they detect it differently. Sequential
    detects it as `_extract_code_block` raising `ValueError` after collecting
    the whole response and finding no fenced block at all. Incremental has no
    equivalent exception -- a response with no opening fence just yields zero
    dispatched statements and finishes as a plain `ok` run -- so
    `_classify_error` recognizes that shape via `_no_code_generated` and
    routes it to the same category explicitly, rather than letting it fall
    through to `Success`.

    Returns:
        One row per (`ModelLabel`, `Strategy`) with a count column for each
        of `Timeout`, `Generation Error`, `SyntaxError`, and `RuntimeError`
        (`_ERROR_CATEGORIES`), 0-filled where a model/strategy had none.
        `RuntimeError` collapses every exception class other than
        `SyntaxError` into one runtime-failure bucket. `Success` runs are
        classified internally (so they aren't miscounted as `RuntimeError`)
        but are not reported as a column here.
    """
    categories = df.apply(_classify_error, axis=1)
    counts = (
        df.assign(ErrorCategory=categories)
        .groupby(["ModelLabel", "Strategy", "ErrorCategory"], sort=False)
        .size()
        .unstack("ErrorCategory", fill_value=0)
    )

    for column in _ERROR_CATEGORIES:
        if column not in counts.columns:
            counts[column] = 0

    breakdown = counts[_ERROR_CATEGORIES].astype(int).reset_index()
    breakdown["Strategy"] = breakdown["Strategy"].str.capitalize()
    breakdown = breakdown.rename(columns={"ModelLabel": "Model"})

    tables_dir.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(tables_dir / "error_breakdown.csv", index=False)
    return breakdown


def _strip_provider_suffix(model_label: str) -> str:
    """Drop the trailing `" (provider)"` suffix from a `ModelLabel` for figure display.

    Args:
        model_label: e.g. `"gemma-3-4b (lmstudio)"`.

    Returns:
        e.g. `"gemma-3-4b"`. Provider stays in the CSVs as table metadata;
        the figures only need the model identity, and dropping it shortens
        the y-axis category labels.
    """
    return model_label.split(" (", 1)[0]


def _stripped_labels(models: Iterable[str]) -> list[str]:
    """Strip the provider suffix off every model label, for x-axis/legend display.

    Args:
        models: `ModelLabel` values, e.g. from `df["ModelLabel"].unique()`.

    Returns:
        One `_strip_provider_suffix()`-ed label per input model, in order.
    """
    return [_strip_provider_suffix(model_label) for model_label in models]


def _bottom_legend(
    fig: Figure, handles: Sequence[object], labels: Sequence[str], ncol: int
) -> None:
    """Place a legend below the axes, explaining a figure's encoding.

    Every figure in this module uses a legend this way instead of an
    in-figure title/annotation, per the no-in-figure-titles /
    decoded-notation requirement -- centralized so that convention (and the
    `loc`/`frameon` it depends on) can't drift between figures.

    Args:
        fig: Figure to attach the legend to.
        handles: Artists to label, in the same order as `labels`.
        labels: One label per handle.
        ncol: Number of legend columns.
    """
    fig.legend(
        handles=handles,
        labels=labels,
        loc="outside lower center",
        ncol=ncol,
        frameon=False,
    )


def _bar_with_iqr(
    ax: Axes,
    x: np.ndarray,
    medians: Sequence[float],
    q1s: Sequence[float],
    q3s: Sequence[float],
    *,
    color: str,
    annotate: Callable[[float], str],
) -> tuple[BarContainer, ErrorbarContainer]:
    """Draw one bar per x position with an asymmetric IQR error bar and a value label.

    Pulled out once `build_speedup_ratio_figure` and
    `build_first_statement_fraction_figure` both needed the identical
    bar/error-bar/annotation mechanics for a per-model median with a
    Q1-Q3 whisker, differing only in color and how the median is formatted.

    Args:
        ax: Axes to draw on.
        x: Bar x-positions, one per model.
        medians: Bar heights.
        q1s: Per-bar lower quartile (error bar's lower whisker).
        q3s: Per-bar upper quartile (error bar's upper whisker, and where
            the value label anchors if it sits higher than the median).
        color: Bar fill color.
        annotate: Formats one median into its value label, e.g.
            `lambda v: f"{v:.2f}x"`.

    Returns:
        The `(bars, error_container)` handles, for the caller's own legend.
    """
    median_arr = np.asarray(medians, dtype=float)
    q1_arr = np.asarray(q1s, dtype=float)
    q3_arr = np.asarray(q3s, dtype=float)

    bars = ax.bar(x, median_arr, color=color, edgecolor="black")
    error_container = ax.errorbar(
        x,
        median_arr,
        yerr=[median_arr - q1_arr, q3_arr - median_arr],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1,
    )
    for xi, median, q3 in zip(x, median_arr, q3_arr, strict=True):
        ax.annotate(
            annotate(median),
            xy=(xi, max(median, q3)),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=FONT_SIZE_PT,
        )
    return bars, error_container


def _speedup_ratio(df: pd.DataFrame, model_label: str, metric: str) -> pd.Series:
    """Per-task speedup ratio (sequential / incremental) for one model and metric.

    Args:
        df: Combined result rows, as returned by `load_results()`.
        model_label: The `ModelLabel` to restrict to.
        metric: Column to pair, e.g. `"FirstExecutedStatement"`.

    Returns:
        `sequential / incremental` per `DatasetRow`, for tasks where both
        strategies have a value for `metric`. A value above 1 means
        incremental was faster on that task.
    """
    wide = _paired_wide(df, model_label, metric)
    return wide["sequential"] / wide["incremental"]


def build_speedup_ratio_figure(df: pd.DataFrame, images_dir: Path = IMAGES_DIR) -> None:
    """Plot median first-statement speedup ratio (sequential / incremental), by model.

    One bar per model, height = median per-task ratio, colored in the
    incremental strategy's palette color. The asymmetric error bar spans the
    IQR (lower whisker = Median - Q1, upper = Q3 - Median). A ratio above the
    dashed 1.0x reference line means incremental reached its first executed
    statement sooner than sequential on the median task. The legend (bar,
    IQR whisker, reference line) sits below the axes rather than as an
    in-figure title/annotation, per the no-in-figure-titles /
    decoded-notation requirement.

    Args:
        df: Combined result rows, as returned by `load_results()`.
        images_dir: Directory the PNG is written into.
    """
    stats: list[tuple[str, float, float, float]] = []
    for model_label in df["ModelLabel"].unique():
        ratio = _speedup_ratio(df, model_label, "FirstExecutedStatement")
        stats.append(
            (model_label, ratio.median(), ratio.quantile(0.25), ratio.quantile(0.75))
        )
    stats.sort(key=lambda item: item[1], reverse=True)
    models, medians, q1s, q3s = zip(*stats, strict=True)
    labels = _stripped_labels(models)

    fig, ax = new_fig(120)

    x = np.arange(1, len(models) + 1)
    bars, error_container = _bar_with_iqr(
        ax,
        x,
        medians,
        q1s,
        q3s,
        color=PALETTE[STRATEGIES[0]],
        annotate=lambda v: f"{v:.2f}×",
    )

    reference_line = ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("Median first-statement\nspeedup, ×\n(Sequential / Incremental)")
    ax.set_ylim(0, max(q3s) + 0.8)

    _bottom_legend(
        fig,
        [bars, error_container, reference_line],
        ["First-statement speedup", "IQR (Q1-Q3)", "No speedup (1.0×)"],
        ncol=2,
    )

    save_fig(fig, "fig-speedup-ratio-barplot.png", images_dir)


def build_first_statement_fraction_figure(
    df: pd.DataFrame, images_dir: Path = IMAGES_DIR
) -> None:
    """Plot where incremental's first statement lands in the total inference timeline.

    Per task, `FirstExecutedStatement / ExecutionTime` as a % -- both are
    wall-clock offsets from the moment the request was sent (see
    `mbpp.main._run_case`'s `started`/`elapsed_seconds`), so this is
    generation and execution combined, not an execution-only metric.
    Sequential is deliberately not plotted here: it cannot execute anything
    until the entire response has been generated, so its share sits at ~100%
    for every model by construction and carries no information. The
    interesting number is how far below 100% incremental's first statement
    lands, and how much that varies by model.

    Args:
        df: Combined result rows, as returned by `load_results()`.
        images_dir: Directory the PNG is written into.
    """
    incremental = df.loc[
        (df["Strategy"] == "incremental")
        & df["FirstExecutedStatement"].notna()
        & df["ExecutionTime"].notna()
    ].copy()
    incremental["FractionPct"] = (
        incremental["FirstExecutedStatement"] / incremental["ExecutionTime"] * 100
    )
    grouped = incremental.groupby("ModelLabel")["FractionPct"]
    medians = grouped.median().sort_values()
    model_order = medians.index
    q1s = grouped.quantile(0.25).reindex(model_order)
    q3s = grouped.quantile(0.75).reindex(model_order)

    labels = _stripped_labels(model_order)
    x = np.arange(1, len(model_order) + 1)

    fig, ax = new_fig(120)

    bars, error_container = _bar_with_iqr(
        ax,
        x,
        medians,
        q1s,
        q3s,
        color=PALETTE["incremental"],
        annotate=lambda v: f"{v:.0f}%",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("First statement position,\n% of total inference time (median)")
    ax.set_ylim(0, 108)

    _bottom_legend(
        fig,
        [bars, error_container],
        ["Median position (incremental)", "IQR (Q1-Q3)"],
        ncol=2,
    )

    save_fig(fig, "fig-first-statement-fraction.png", images_dir)


def main() -> None:
    df = load_results()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(df)} rows from {RESULTS_DIR}")

    build_timing_summary(df)
    build_significance_table(df)
    build_pass_correctness(df)
    build_token_usage(df)
    build_error_breakdown(df)

    build_speedup_ratio_figure(df)
    build_first_statement_fraction_figure(df)


if __name__ == "__main__":
    main()
