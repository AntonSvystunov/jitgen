# Paper figures for `iptc-bfcl-report`; needs the `analysis` dependency group.
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # after selecting the backend
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter, PercentFormatter

from iptc_bfcl.report import (
    ARMS,
    LANGUAGES,
    Results,
    Run,
    bootstrap_median_interval,
    model_label,
    paired_differences,
    wilson_interval,
)

# The reference palette's categorical slots, in its fixed order. Adjacent
# slots pass the colour-blindness checks; only the first three pass them as
# all pairs, so a scatter of more than three models is split into panels.
# Models keep their slot in every figure (assigned by first appearance in the
# results), and a marker shape per model is the secondary encoding the
# low-contrast slots (aqua, yellow, magenta) require.
MODEL_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
# Most models one scatter panel may show (see `MODEL_COLORS`).
MAX_MODELS_PER_SCATTER = 3
# Time components are neutral, so they never read as a model.
GENERATION_LIGHT = "#d6d4cc"
GENERATION_DARK = "#a3a198"
TOOL_DARK = "#3d3c39"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3de"
SINGLE_COLUMN = 3.4
DOUBLE_COLUMN = 7.0

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _style(model_index: int) -> dict[str, str]:
    return {
        "color": MODEL_COLORS[model_index % len(MODEL_COLORS)],
        "marker": MODEL_MARKERS[model_index % len(MODEL_MARKERS)],
    }


def _grid(ax: Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig: Figure, directory: Path, name: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{name}.pdf", directory / f"{name}.png"]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def _model_legend(ax: Axes, models: list[str], **kwargs: object) -> None:
    handles = [
        Line2D(
            [], [], linestyle="none", markersize=5, label=model_label(m), **_style(i)
        )
        for i, m in enumerate(models)
    ]
    ax.legend(handles=handles, **kwargs)


def accuracy_figure(results: Results, directory: Path) -> list[Path]:
    """Accuracy per arm and model, with 95% Wilson intervals."""
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.4))
    arms = list(ARMS)
    offsets = [
        0.22 * (i - (len(results.models) - 1) / 2) for i in range(len(results.models))
    ]
    for i, model in enumerate(results.models):
        for row, arm in enumerate(arms):
            runs = results.by(model, arm)
            if not runs:
                continue
            k, n = sum(r.correct for r in runs), len(runs)
            low, high = wilson_interval(k, n)
            y = row + offsets[i]
            ax.plot(
                [100 * low, 100 * high],
                [y, y],
                color=_style(i)["color"],
                linewidth=1.2,
                alpha=0.6,
            )
            ax.plot(
                100 * k / n,
                y,
                markersize=5,
                linestyle="none",
                markeredgecolor="white",
                markeredgewidth=0.6,
                **_style(i),
            )
    ax.set_yticks(range(len(arms)), [ARMS[a] for a in arms])
    ax.invert_yaxis()
    ax.set_xlim(-3, 100)
    ax.set_xlabel("Correct runs (%), 95% Wilson interval")
    _grid(ax)
    _model_legend(
        ax,
        results.models,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=len(results.models),
        handletextpad=0.2,
        columnspacing=1.0,
    )
    return _save(fig, directory, "accuracy")


def time_breakdown_figure(results: Results, directory: Path) -> list[Path]:
    """Mean run time split into generation and what came after it."""
    models = results.models
    columns = len(models) if len(models) <= 3 else 2
    rows = -(-len(models) // columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(DOUBLE_COLUMN, 1.9 * rows),
        sharey=True,
        squeeze=False,
    )
    arms = list(ARMS)
    for ax in axes.flat[len(models) :]:
        ax.set_visible(False)
    for ax, model in zip(axes.flat, models):
        for row, arm in enumerate(arms):
            runs = results.by(model, arm)
            if not runs:
                continue
            generation = statistics.mean(r.generation for r in runs)
            after = statistics.mean(r.execution for r in runs)
            overlap = statistics.mean(r.overlap for r in runs)
            ax.barh(row, generation, color=GENERATION_LIGHT, height=0.62)
            if overlap > 0:
                # The part of generation tool calls ran under (IPTC's gain).
                ax.barh(
                    row,
                    overlap,
                    left=generation - overlap,
                    color=GENERATION_DARK,
                    height=0.62,
                )
            ax.barh(
                row,
                after,
                left=generation,
                color=TOOL_DARK,
                height=0.62,
                edgecolor="white",
                linewidth=1,
            )
            ax.text(
                generation + after,
                row,
                f" {generation + after:.2f}",
                va="center",
                fontsize=7,
                color=INK_SECONDARY,
            )
        ax.set_title(model_label(model), loc="left")
        ax.set_xlabel("Mean seconds per run")
        ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
        _grid(ax)
    for row_axes in axes:
        row_axes[0].set_yticks(range(len(arms)), [ARMS[a] for a in arms])
    axes[0][0].invert_yaxis()
    fig.legend(
        handles=[
            Patch(color=GENERATION_LIGHT, label="Generation"),
            Patch(
                color=GENERATION_DARK,
                label="Generation with tool calls running (overlap)",
            ),
            Patch(color=TOOL_DARK, label="After generation (tools, execution)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
    )
    fig.tight_layout()
    return _save(fig, directory, "time_breakdown")


def paired_figure(results: Results, directory: Path) -> list[Path]:
    """Per-entry saving of IPTC over PTC, with the median and its 95% CI."""
    measures = [
        ("Wall time", lambda r: r.wall),
        ("Wall time − generation", lambda r: r.execution),
    ]
    rows = [(m, lang) for m in results.models for lang in LANGUAGES]
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COLUMN, 0.32 * len(rows) + 0.9), sharey=True
    )
    jitter = random.Random(0)
    for ax, (label, measure) in zip(axes, measures):
        for y, (model, language) in enumerate(rows):
            diffs = paired_differences(results, model, language, measure)
            if not diffs:
                continue
            style = _style(results.models.index(model))
            ax.scatter(
                diffs,
                [y + jitter.uniform(-0.18, 0.18) for _ in diffs],
                s=10,
                color=style["color"],
                marker=style["marker"],
                alpha=0.55,
                linewidths=0,
            )
            low, high = bootstrap_median_interval(diffs)
            ax.plot([low, high], [y, y], color=INK, linewidth=1.4)
            ax.plot(
                statistics.median(diffs),
                y,
                marker="|",
                markersize=9,
                markeredgewidth=1.6,
                color=INK,
            )
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.set_title(label, loc="left")
        ax.set_xlabel("PTC − IPTC (s);  > 0 means IPTC was faster")
        _grid(ax)
    axes[0].set_yticks(
        range(len(rows)), [f"{model_label(m)} · {LANGUAGES[lang]}" for m, lang in rows]
    )
    axes[0].invert_yaxis()
    handles = [
        Line2D(
            [],
            [],
            color=INK,
            linewidth=1.4,
            marker="|",
            markersize=8,
            markeredgewidth=1.6,
            label="Median, 95% bootstrap CI",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=4,
            color=INK_SECONDARY,
            alpha=0.6,
            label="One entry",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    return _save(fig, directory, "paired_savings")


def _hidden_share(results: Results, model: str) -> tuple[list[float], list[float]]:
    """Per IPTC run with 2+ calls: time to write one call, share of tool time hidden."""
    runs = [
        r
        for arm in ("iptc-python", "iptc-javascript")
        for r in results.by(model, arm)
        if r.argument_seconds and r.calls >= 2 and r.tool > 0
    ]
    return (
        [r.argument_seconds / r.calls for r in runs],
        [min(1.0, r.overlap / r.tool) for r in runs],
    )


def _overlap_axes(ax: Axes, tool_delay: float | None) -> None:
    if tool_delay:
        ax.axvline(tool_delay, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
        ax.text(
            tool_delay,
            1.04,
            f" tool latency {tool_delay:g} s",
            fontsize=7,
            color=INK_SECONDARY,
            va="bottom",
        )
    ax.set_xscale("log")
    low, high = ax.get_xlim()
    ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2) if low <= t <= high]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.03, 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _grid(ax, "both")


def overlap_figure(
    results: Results, directory: Path, tool_delay: float | None
) -> list[Path]:
    """Share of tool time IPTC hid, against the time the model took per call.

    Up to three models share one panel. With more, every model gets its own
    panel, with the other models' runs in grey behind it for reference.
    """
    models = results.models
    points = {model: _hidden_share(results, model) for model in models}
    x_label = "Time the model took to write one call (s, log)"
    y_label = "Tool time hidden in generation"
    if len(models) <= MAX_MODELS_PER_SCATTER:
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.4))
        for i, model in enumerate(models):
            ax.scatter(
                *points[model],
                s=14,
                alpha=0.7,
                linewidths=0.4,
                edgecolors="white",
                label=model_label(model),
                **_style(i),
            )
        _overlap_axes(ax, tool_delay)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc="upper left", handletextpad=0.2)
        return _save(fig, directory, "overlap_mechanism")
    columns = 2
    rows = -(-len(models) // columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(DOUBLE_COLUMN * 0.75, 2.0 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax in axes.flat[len(models) :]:
        ax.set_visible(False)
    for i, (ax, model) in enumerate(zip(axes.flat, models)):
        for other in models:
            if other != model:
                ax.scatter(*points[other], s=8, color=GRID, linewidths=0)
        ax.scatter(
            *points[model],
            s=14,
            alpha=0.8,
            linewidths=0.4,
            edgecolors="white",
            **_style(i),
        )
        ax.set_title(model_label(model), loc="left")
        _overlap_axes(ax, tool_delay)
    fig.supxlabel(x_label, fontsize=8)
    fig.supylabel(y_label, fontsize=8)
    fig.tight_layout()
    return _save(fig, directory, "overlap_mechanism")


def _example(results: Results) -> tuple[Run, Run] | None:
    """A PTC/IPTC pair with identical generations and the most calls."""
    best: tuple[float, Run, Run] | None = None
    for model in results.models:
        ptc = {r.entry_id: r for r in results.by(model, "ptc-python")}
        for iptc in results.by(model, "iptc-python"):
            other = ptc.get(iptc.entry_id)
            if (
                other is None
                or other.output_tokens != iptc.output_tokens
                or iptc.calls < 2
            ):
                continue
            score = iptc.calls * 10 + (other.wall - iptc.wall)
            if best is None or score > best[0]:
                best = (score, other, iptc)
    return (best[1], best[2]) if best else None


def _run_rows(
    results: Results, run: Run
) -> tuple[dict[str, str], list[dict[str, str]]]:
    strategy, language = run.arm.split("-")
    key = (run.model, run.entry_id, strategy, language)
    turn = next(
        t
        for t in results.turns
        if (t["model"], t["entry_id"], t["strategy"], t["language"]) == key
        and t["index"] == "0"
    )
    calls = [
        c
        for c in results.tool_calls
        if (c["model"], c["entry_id"], c["strategy"], c["language"]) == key
    ]
    return turn, calls


def timeline_figure(results: Results, directory: Path) -> list[Path]:
    """One response streamed and executed by PTC and by IPTC, on one time axis."""
    pair = _example(results)
    if pair is None:
        return []
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 1.7))
    labels = []
    for row, run in enumerate(pair):
        turn, calls = _run_rows(results, run)
        start = float(turn["request_start"])
        first_args = float(turn["first_arguments"]) - start
        last_args = float(turn["last_arguments"]) - start
        y = row * 1.6
        ax.barh(y, first_args, height=0.5, color=GENERATION_LIGHT)
        ax.barh(
            y,
            last_args - first_args,
            left=first_args,
            height=0.5,
            color=GENERATION_DARK,
        )
        for call in calls:
            begin, end = float(call["start"]) - start, float(call["end"]) - start
            ax.barh(
                y - 0.55,
                end - begin,
                left=begin,
                height=0.38,
                color=TOOL_DARK,
                edgecolor="white",
                linewidth=0.8,
            )
        ax.plot([run.wall, run.wall], [y - 0.8, y + 0.3], color=INK, linewidth=1)
        ax.text(run.wall, y + 0.32, f" {run.wall:.2f} s", fontsize=7, va="bottom")
        labels.append((y - 0.25, ARMS[run.arm]))
    ax.set_yticks([y for y, _ in labels], [label for _, label in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Seconds since the request")
    ax.set_title(
        f"{model_label(pair[1].model)} · {pair[1].entry_id} · {pair[1].calls} calls "
        "(identical generation in both arms)",
        loc="left",
    )
    _grid(ax)
    ax.legend(
        handles=[
            Patch(color=GENERATION_LIGHT, label="Generation before the code"),
            Patch(color=GENERATION_DARK, label="Writing the code"),
            Patch(color=TOOL_DARK, label="Tool call"),
            Line2D([], [], color=INK, linewidth=1, label="Run done"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
    )
    return _save(fig, directory, "timeline_example")


def render_figures(results: Results, directory: Path) -> list[Path]:
    """Render every figure as PDF (for the paper) and PNG (for previews).

    Args:
        results: The loaded results.
        directory: Where the figures go.

    Returns:
        The files written.
    """
    delays = {float(t["tool_delay"]) for t in results.turns if t.get("tool_delay")}
    tool_delay = next(iter(delays)) if len(delays) == 1 else None
    return [
        *timeline_figure(results, directory),
        *paired_figure(results, directory),
        *overlap_figure(results, directory, tool_delay),
        *time_breakdown_figure(results, directory),
        *accuracy_figure(results, directory),
    ]
