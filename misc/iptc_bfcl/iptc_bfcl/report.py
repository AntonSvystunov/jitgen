import argparse
import csv
import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Display order and labels; the keys are `strategy-language` as in the CSVs.
ARMS = {
    "baseline": "Baseline",
    "ptc-python": "PTC (Py)",
    "iptc-python": "IPTC (Py)",
    "ptc-javascript": "PTC (JS)",
    "iptc-javascript": "IPTC (JS)",
}
LANGUAGES = {"python": "Python", "javascript": "JavaScript"}


@dataclass
class Run:
    """One row of `runs.csv`, plus its first turn's argument-streaming span."""

    model: str
    entry_id: str
    category: str
    arm: str
    correct: bool
    status: str
    grade_reason: str
    wall: float
    generation: float
    tool: float
    overlap: float
    calls: int
    output_tokens: int
    argument_seconds: float | None = None

    @property
    def execution(self) -> float:
        """Time outside generation: tools and execution the model didn't hide."""
        return self.wall - self.generation


@dataclass
class Results:
    """Everything the report reads from one results directory."""

    runs: list[Run]
    turns: list[dict[str, str]]
    tool_calls: list[dict[str, str]]
    models: list[str] = field(default_factory=list)

    def by(self, model: str, arm: str) -> list[Run]:
        return [r for r in self.runs if r.model == model and r.arm == arm]


def model_label(model: str) -> str:
    """`lmstudio:qwen/qwen3.6-27b` -> `qwen3.6-27b`."""
    return model.split(":", 1)[-1].rsplit("/", 1)[-1]


def _arm(row: dict[str, str]) -> str:
    language = row["language"]
    return f"{row['strategy']}-{language}" if language else row["strategy"]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_results(directory: Path, exclude: Sequence[str] = ()) -> Results:
    """Load a results directory, dropping models whose id contains an excluded text.

    Args:
        directory: A `--out` directory of `iptc-bfcl`.
        exclude: Substrings of model ids to leave out (e.g. a throttled model).

    Returns:
        The runs, turns and tool calls of the kept models.
    """

    def kept(row: dict[str, str]) -> bool:
        return not any(text in row["model"] for text in exclude)

    turns = [t for t in _read(directory / "turns.csv") if kept(t)]
    first_turns = {
        (t["model"], t["entry_id"], _arm(t), t["tool_delay"]): t
        for t in turns
        if t["index"] == "0"
    }
    runs = []
    for row in _read(directory / "runs.csv"):
        if not kept(row):
            continue
        turn = first_turns.get(
            (row["model"], row["entry_id"], _arm(row), row["tool_delay"]), {}
        )
        argument_seconds = (
            float(turn["last_arguments"]) - float(turn["first_arguments"])
            if turn.get("first_arguments") and turn.get("last_arguments")
            else None
        )
        runs.append(
            Run(
                model=row["model"],
                entry_id=row["entry_id"],
                category=row["category"],
                arm=_arm(row),
                correct=row["correct"] == "True",
                status=row["status"],
                grade_reason=row["grade_reason"],
                wall=float(row["wall_seconds"]),
                generation=float(row["model_seconds"]),
                tool=float(row["tool_seconds"]),
                overlap=float(row["overlap_seconds"]),
                calls=int(row["tool_calls"]),
                output_tokens=int(row["est_output_tokens"]),
                argument_seconds=argument_seconds,
            )
        )
    # First appearance, not alphabetical: a model added to the results later
    # never shifts the others' colours or places.
    models = list(dict.fromkeys(r.model for r in runs))
    calls = [c for c in _read(directory / "tool_calls.csv") if kept(c)]
    return Results(runs, turns, calls, models)


# --- statistics ---------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a proportion, sound for small `n`.

    Args:
        successes: Number of successes.
        n: Number of trials.
        z: The normal quantile of the confidence level.

    Returns:
        The interval's bounds, `(0, 1)` when `n` is 0.
    """
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_median_interval(
    values: Sequence[float], *, resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile-bootstrap interval of the median.

    Args:
        values: The sample, e.g. per-entry paired differences.
        resamples: Bootstrap resamples.
        seed: Seeds the resampling, so the report is reproducible.

    Returns:
        The interval's bounds.

    Raises:
        ValueError: If `values` is empty.
    """
    if not values:
        msg = "no values to bootstrap"
        raise ValueError(msg)
    rng = random.Random(seed)
    n = len(values)
    medians = sorted(
        statistics.median(rng.choices(values, k=n)) for _ in range(resamples)
    )
    return medians[int(0.025 * resamples)], medians[int(0.975 * resamples) - 1]


def sign_test(differences: Sequence[float]) -> float:
    """Exact two-sided sign test that positive and negative differences are equally likely.

    Ties (zero differences) are dropped, as is standard.

    Args:
        differences: Paired differences.

    Returns:
        The p-value, 1.0 when every difference is zero.
    """
    positive = sum(1 for d in differences if d > 0)
    n = sum(1 for d in differences if d != 0)
    if n == 0:
        return 1.0
    tail = min(positive, n - positive)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / 2**n
    return min(1.0, 2 * probability)


@dataclass(frozen=True)
class PairedComparison:
    """IPTC against PTC on the same entries, for one model and language."""

    model: str
    language: str
    pairs: int
    identical_outputs: int
    wall_median: float
    wall_interval: tuple[float, float]
    wall_faster: int
    wall_p: float
    execution_median: float
    execution_interval: tuple[float, float]
    execution_faster: int
    execution_p: float


def paired_comparisons(results: Results) -> list[PairedComparison]:
    """Compare IPTC with PTC entry by entry, per model and language.

    Differences are PTC minus IPTC, so positive means IPTC was faster.
    "Execution" is wall time minus generation time: what the arm spent beyond
    the model's own output, which is where IPTC's effect lives and which the
    run-to-run variation in generation length doesn't touch.

    Args:
        results: The loaded results.

    Returns:
        One comparison per model and language with at least one pair.
    """
    comparisons = []
    for model in results.models:
        for language in LANGUAGES:
            ptc = {r.entry_id: r for r in results.by(model, f"ptc-{language}")}
            iptc = {r.entry_id: r for r in results.by(model, f"iptc-{language}")}
            entries = sorted(ptc.keys() & iptc.keys())
            if not entries:
                continue
            wall = [ptc[e].wall - iptc[e].wall for e in entries]
            execution = [ptc[e].execution - iptc[e].execution for e in entries]
            comparisons.append(
                PairedComparison(
                    model=model,
                    language=language,
                    pairs=len(entries),
                    identical_outputs=sum(
                        ptc[e].output_tokens == iptc[e].output_tokens for e in entries
                    ),
                    wall_median=statistics.median(wall),
                    wall_interval=bootstrap_median_interval(wall),
                    wall_faster=sum(d > 0 for d in wall),
                    wall_p=sign_test(wall),
                    execution_median=statistics.median(execution),
                    execution_interval=bootstrap_median_interval(execution),
                    execution_faster=sum(d > 0 for d in execution),
                    execution_p=sign_test(execution),
                )
            )
    return comparisons


def paired_differences(
    results: Results, model: str, language: str, measure: Callable[[Run], float]
) -> list[float]:
    """PTC minus IPTC of `measure`, per entry both arms ran."""
    ptc = {r.entry_id: r for r in results.by(model, f"ptc-{language}")}
    iptc = {r.entry_id: r for r in results.by(model, f"iptc-{language}")}
    return [
        measure(ptc[e]) - measure(iptc[e]) for e in sorted(ptc.keys() & iptc.keys())
    ]


# --- tables -------------------------------------------------------------------


@dataclass
class Table:
    """A table to write as Markdown, LaTeX (booktabs) and CSV."""

    name: str
    caption: str
    header: list[str]
    rows: list[list[str]]
    notes: str = ""

    def markdown(self) -> str:
        lines = [
            f"**{self.caption}**",
            "",
            "| " + " | ".join(self.header) + " |",
            "|" + "|".join("---" for _ in self.header) + "|",
            *("| " + " | ".join(row) + " |" for row in self.rows),
        ]
        if self.notes:
            lines += ["", self.notes]
        return "\n".join(lines) + "\n"

    def latex(self) -> str:
        def escape(text: str) -> str:
            for old, new in [
                ("\\", r"\textbackslash{}"),
                ("%", r"\%"),
                ("_", r"\_"),
                ("&", r"\&"),
                ("#", r"\#"),
            ]:
                text = text.replace(old, new)
            return text.replace("–", "--").replace("−", "$-$").replace("×", r"$\times$")

        columns = "l" + "r" * (len(self.header) - 1)
        body = [" & ".join(escape(cell) for cell in row) + r" \\" for row in self.rows]
        notes = (
            f"\n\\par\\smallskip\\footnotesize {escape(self.notes)}"
            if self.notes
            else ""
        )
        return "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering\small",
                f"\\caption{{{escape(self.caption)}}}",
                f"\\label{{tab:{self.name}}}",
                f"\\begin{{tabular}}{{{columns}}}",
                r"\toprule",
                " & ".join(escape(h) for h in self.header) + r" \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular}" + notes,
                r"\end{table}",
                "",
            ]
        )

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{self.name}.md").write_text(self.markdown(), encoding="utf-8")
        (directory / f"{self.name}.tex").write_text(self.latex(), encoding="utf-8")
        with (directory / f"{self.name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(self.header)
            writer.writerows(self.rows)


def _seconds(value: float) -> str:
    return f"{value:.2f}"


def _signed(value: float) -> str:
    return f"{value:+.2f}".replace("-", "−")


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def accuracy_table(results: Results) -> Table:
    """Correct runs per model and arm, with 95% Wilson intervals."""
    rows = []
    for model in results.models:
        row = [model_label(model)]
        for arm in ARMS:
            runs = results.by(model, arm)
            k, n = sum(r.correct for r in runs), len(runs)
            low, high = wilson_interval(k, n)
            row.append(
                f"{k}/{n} ({100 * k / n:.0f}%) [{100 * low:.0f}–{100 * high:.0f}]"
                if n
                else "–"
            )
        rows.append(row)
    return Table(
        "accuracy",
        "Accuracy (BFCL AST checker) by model and arm",
        ["Model", *ARMS.values()],
        rows,
        "Correct/total (%), with 95% Wilson interval in brackets.",
    )


def latency_table(results: Results) -> Table:
    """Median time components per model and arm."""
    rows = []
    for model in results.models:
        for arm, label in ARMS.items():
            runs = results.by(model, arm)
            if not runs:
                continue

            components = [
                [r.wall for r in runs],
                [r.generation for r in runs],
                [r.tool for r in runs],
                [r.overlap for r in runs],
                [r.execution for r in runs],
            ]
            rows.append(
                [
                    model_label(model),
                    label,
                    str(len(runs)),
                    *(_seconds(statistics.median(values)) for values in components),
                ]
            )
    return Table(
        "latency",
        "Median time per run (seconds) by model and arm",
        [
            "Model",
            "Arm",
            "n",
            "Wall",
            "Generation",
            "Tool",
            "Overlap",
            "Wall − generation",
        ],
        rows,
        "Generation: request to the model's last token. Tool: union of tool-call "
        "time. Overlap: tool time spent while the model was still generating.",
    )


def paired_table(comparisons: list[PairedComparison]) -> Table:
    """IPTC against PTC, paired by entry."""
    rows = [
        [
            model_label(c.model),
            LANGUAGES[c.language],
            str(c.pairs),
            f"{c.identical_outputs}/{c.pairs}",
            f"{_signed(c.wall_median)} [{_signed(c.wall_interval[0])}, {_signed(c.wall_interval[1])}]",
            f"{c.wall_faster}/{c.pairs}",
            _p(c.wall_p),
            f"{_signed(c.execution_median)} [{_signed(c.execution_interval[0])}, {_signed(c.execution_interval[1])}]",
            f"{c.execution_faster}/{c.pairs}",
            _p(c.execution_p),
        ]
        for c in comparisons
    ]
    return Table(
        "paired",
        "IPTC vs. PTC, paired by entry (PTC − IPTC, seconds; positive = IPTC faster)",
        [
            "Model",
            "Language",
            "Pairs",
            "Identical output",
            "Δ wall, median [95% CI]",
            "IPTC faster",
            "p",
            "Δ (wall − generation), median [95% CI]",
            "IPTC faster",
            "p",
        ],
        rows,
        "CI: percentile bootstrap of the median (10,000 resamples). p: exact "
        "two-sided sign test. Identical output: both arms produced the same "
        "number of output tokens, i.e. the same generation.",
    )


def failure_table(results: Results) -> Table:
    """Wrong runs per entry, out of the arms each model ran."""
    failing = sorted({r.entry_id for r in results.runs if not r.correct})
    rows = []
    for entry in failing:
        row = [entry]
        for model in results.models:
            runs = [r for r in results.runs if r.model == model and r.entry_id == entry]
            row.append(
                f"{sum(not r.correct for r in runs)}/{len(runs)}" if runs else "–"
            )
        rows.append(row)
    return Table(
        "failures",
        "Entries with at least one wrong run (wrong arms / arms run)",
        ["Entry", *(model_label(m) for m in results.models)],
        rows,
    )


def write_report(results: Results, out: Path, *, figures: bool = True) -> list[Path]:
    """Write every table, and the figures unless `figures` is false.

    Args:
        results: The loaded results.
        out: The report directory.
        figures: Also render the figures (needs the `analysis` group).

    Returns:
        The files written.
    """
    comparisons = paired_comparisons(results)
    tables = [
        accuracy_table(results),
        latency_table(results),
        paired_table(comparisons),
        failure_table(results),
    ]
    for table in tables:
        table.write(out / "tables")
    (out / "tables.md").write_text(
        "\n".join(t.markdown() for t in tables), encoding="utf-8"
    )
    written = sorted((out / "tables").iterdir()) + [out / "tables.md"]
    if figures:
        # Imported here so the tables never need matplotlib.
        from iptc_bfcl.figures import render_figures

        written += render_figures(results, out / "figures")
    return written


def main(argv: list[str] | None = None) -> None:
    """Entry point for `iptc-bfcl-report`.

    Args:
        argv: Command-line arguments; `sys.argv` when `None`.
    """
    parser = argparse.ArgumentParser(
        prog="iptc-bfcl-report",
        description="Paper tables and figures from an iptc-bfcl results directory.",
    )
    parser.add_argument("results", type=Path, help="an iptc-bfcl --out directory")
    parser.add_argument(
        "--exclude",
        type=lambda value: [v.strip() for v in value.split(",") if v.strip()],
        default=[],
        help="comma-separated substrings of model ids to leave out",
    )
    parser.add_argument("--out", type=Path, help="defaults to <results>/report")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    results = load_results(args.results, args.exclude)
    if not results.runs:
        raise SystemExit(f"no runs in {args.results}")
    out = args.out or args.results / "report"
    for path in write_report(results, out, figures=not args.no_figures):
        print(path)


if __name__ == "__main__":
    main()
