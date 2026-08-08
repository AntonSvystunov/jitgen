# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

JitGen implements **grammar-guided incremental execution of LLM-generated code**: instead of waiting for a model to finish generating a code block before executing it (generate-then-execute), JitGen parses the growing output buffer against the target language's grammar and dispatches each top-level statement to a persistent interpreter as soon as its boundary is provably fixed — i.e. as soon as no future token could still change it. This lowers time-to-first-output and surfaces syntax/runtime errors as early as the offending prefix is complete, without touching model decoding.

**Read `docs/METHOD.md` before making any change to the core loop** (extraction, session buffering, dispatch/flush timing). It is the formal spec: numbered principles (P1–P11), preconditions, a glossary of exact terminology to reuse in code/comments/PRs, and a review checklist. Code in `libs/jitgen` is the reference implementation of that spec — if a change seems to conflict with a principle there, the spec wins; open the question rather than silently deviating.

The repo is a monorepo shell: currently the only package is `libs/jitgen`. `misc/` is present but currently empty in this checkout — it previously held paper/evaluation artifacts (notebooks, plots, result CSVs) that were removed; don't assume anything lives there without checking.

## State of the code (read before assuming an API exists)

What exists today is the **core protocol layer plus one concrete segmenter** — there is currently no bundled statement extractor (Lark-based Python grammar) or executor (in-process Python REPL) implementation; `jitgen/extractors/` and `jitgen/executors/` (besides the `ExecutorBase` ABC) are empty. Any `StatementExtractor` or `BaseExecutor` needed for a working end-to-end pipeline must be supplied by the caller against the protocols in `jitgen/base.py`. Do not assume a `create_python_session()`-style factory exists — check first. Because of this, `tests/unit_tests/test_session.py` and `test_driver.py` test `Session`/`StreamDriver` against small hand-written `StatementExtractor`/`BaseExecutor` doubles defined in `tests/unit_tests/conftest.py` (`FakeExtractor`, `FakeExecutor`), not a real grammar or interpreter — reuse those fixtures rather than inventing new doubles when extending these tests.

## Commands

Run all commands from `libs/jitgen/` (the package root; `pyproject.toml` lives here, not at the repo root — there's no root-level `pyproject.toml` for the monorepo).

```bash
uv sync --group test         # install test dependencies
uv run pytest                # run the full test suite
uv run pytest tests/unit_tests/test_session.py           # single file
uv run pytest tests/unit_tests/test_session.py::test_name  # single test
uv run pytest --cov=jitgen   # with coverage (coverage config omits tests/*)
```

Requires Python >=3.13. `pyproject.toml` sets `asyncio_mode = "auto"`, so async test functions run without needing `@pytest.mark.asyncio` — relevant here since the whole public surface (`Session`, `StreamDriver`, `ExecutorBase`) is async.

**Fragile bit:** `pyproject.toml` has no `[build-system]` table, so `uv sync` treats `jitgen` as a virtual project (`source = { virtual = "." }` in `uv.lock`) and never installs it into `.venv` — `uv pip list` shows no `jitgen` entry after `uv sync`. `import jitgen` inside tests works anyway, but only because `tests/__init__.py` *and* `tests/unit_tests/__init__.py` both exist: with that package chain in place, pytest's default (`prepend`) import mode inserts the directory *above* `tests/` — i.e. `libs/jitgen`, which also contains the `jitgen/` source — onto `sys.path`. Losing either `__init__.py`, or adding a test subdirectory without one, silently breaks `import jitgen` again (`ModuleNotFoundError`) without touching any application code. If that recurs, the durable fix is a `[build-system]`/`[tool.hatch.build]` section so the package installs properly instead of relying on this layout quirk.

`ruff` is not a declared project dependency (`uv run ruff` fails), but it's the required linter/formatter — run it via `uvx` (downloads and runs it in an ephemeral environment, no project changes needed) after editing any Python file:

```bash
uvx ruff format .        # format
uvx ruff check --fix .   # autofix (import ordering, etc.)
```

This matches what VS Code already does on save via the `charliermarsh.ruff` extension (`.vscode/settings.json`).

## Architecture

The library is a small pipeline of independently-swappable pieces, each defined as a `Protocol` in `jitgen/base.py` so grammar, framing, and execution backend can vary independently:

- **`CodeSegmenter`** (`jitgen/segmenters/`) — pulls raw code text out of a raw model stream one chunk at a time; owns whatever surrounding framing the model uses (markdown fences, tool-call argument deltas, etc). `MarkerSegmenter` / `markdown_code()` in `marker.py` is the one shipped implementation, matching a start/end string pair and handling markers split across chunk boundaries. Segmenter concerns are entirely separate from grammar concerns: the same extractor and executor work regardless of framing.
- **`StatementExtractor`** (`jitgen/extractors/`, currently unimplemented) — the grammar-aware half. Turns an evolving source buffer into `(ready_statements, leftover_buffer)`, deciding which parse errors mean "incomplete, need more input" vs. "unrecoverable" (raises `SyntaxError` for the latter). This is where P1/P3 (boundaries come from the grammar; a boundary is fixed only when self-delimiting) get enforced for a specific language.
- **`BaseExecutor`** / **`ExecutorBase`** (`jitgen/executors/`) — async executor contract: `aexecute`, `acancel`, `aclose`. Subclass `ExecutorBase` (an ABC) to get no-op `acancel`/`aclose` for free and only implement `aexecute`. State must persist across calls within one session (P5) — this is meant to behave like a REPL, not a fresh interpreter per statement.
- **`Session`** (`session.py`) — the grammar-agnostic, executor-agnostic core loop. `push()` is synchronous, appends to an internal buffer, and — as a cheap pre-filter before invoking the (expensive) extractor — only actually calls `extract()` when the appended span could plausibly have closed a statement (a newline, or a non-blank char at column 0; see `_may_close_statement`). Extracted statements are queued and run on a background `asyncio.Task` worker so `push()` never blocks the caller reading the model stream. Tracks `SessionStats` (first/last statement timing, counts) and enforces the buffer invariant (P6): the buffer always holds exactly the unexecuted suffix. `reset()` cancels in-flight work and clears state but preserves stats and executor REPL state across turns; `aclose()` tears the worker (and optionally the executor) down.
- **`StreamDriver`** (`driver.py`) — wires a `Session` to a `CodeSegmenter` and owns the fetch-parse-dispatch loop consumers would otherwise reimplement: `apush(chunk)` feeds one raw chunk through the segmenter into the session and returns whatever stdout is ready; `afinish()` flushes an unterminated trailing block (P9) and raises the first error encountered anywhere in the stream; `areset()` clears both halves between turns without dropping executor state.
- **`errors.py`** — `JitGenError` is the common base (carries `has_timed_out`/`has_cancelled` so callers don't need to know the subclass to branch on them, plus `partial_message` for integrations that need to record what the model had emitted when generation was halted). `ExtractionError` wraps unrecoverable parse failures; `ExecutionError` wraps runtime failures, with `from_result()` to build one from a failed `ExecutionResult` while preserving the executor's own error string verbatim.

### Where the METHOD.md principles show up in code

Concrete mechanisms — not restatements — for how `docs/METHOD.md` §3's principles are enforced, so a change doesn't accidentally undo one:

- **P2** (never dispatch the last statement while streaming): `Session._extract` (called from `push`) and the flush path (`StreamDriver.afinish` + segmenter `finalize()`, called from `Session.result`) are deliberately separate code paths rather than one function with a flag.
- **P6/P8** (buffer invariant; first error halts everything): a `reset()` bumps an internal generation counter, and work items / in-flight execution tagged with a stale generation are discarded rather than allowed to mutate state for the *next* turn (`_WorkItem.generation`, checked in `_worker_loop` and `_execute`).
- `push()`/`apush()` never raise for extraction errors mid-stream — they set `has_error`/`session.error` so the caller can poll and abort the *model* stream on its own schedule, and the exception surfaces later from `result()`/`afinish()`. The one case called out in `StreamDriver`'s docstring where output is due immediately rather than on the next poll: a block that closes cleanly mid-stream.

## Code quality standards

All Python code MUST include type hints and return types.

```python title="Example"
def filter_unknown_users(users: list[str], known_users: set[str]) -> list[str]:
    """Single line description of the function.

    Any additional context about the function can go here.

    Args:
        users: List of user identifiers to filter.
        known_users: Set of known/valid user identifiers.

    Returns:
        List of users that are not in the `known_users` set.
    """
```

- Use descriptive, self-explanatory variable names.
- Follow existing patterns in the codebase you're modifying.
- Attempt to break up complex functions (>20 lines) into smaller, focused functions where it makes sense.

## Testing requirements

Every new feature or bugfix MUST be covered by unit tests.

- Unit tests: `tests/unit_tests/` (no network calls allowed)
- Integration tests: `tests/integration_tests/` (network calls permitted)
- Use `pytest` as the testing framework; if in doubt, check other existing tests for examples.
- The testing file structure should mirror the source code structure.

**Checklist:**

- [ ] Tests fail when your new logic is broken
- [ ] Happy path is covered
- [ ] Edge cases and error conditions are tested
- [ ] Use fixtures/mocks for external dependencies
- [ ] Does the test suite fail if your new logic is broken?

## Security and risk assessment

- Proper exception handling (no bare `except:`) and use a `msg` variable for error messages
- Remove unreachable/commented code before committing
- Race conditions or resource leaks (file handles, sockets, threads).
- Ensure proper resource cleanup (file handles, connections)

## Documentation standards

Use Google-style docstrings with Args section for all public functions.

```python title="Example"
def send_email(to: str, msg: str, *, priority: str = "normal") -> bool:
    """Send an email to a recipient with specified priority.

    Any additional context about the function can go here.

    Args:
        to: The email address of the recipient.
        msg: The message body to send.
        priority: Email priority level.

    Returns:
        `True` if email was sent successfully, `False` otherwise.

    Raises:
        InvalidEmailError: If the email address format is invalid.
        SMTPConnectionError: If unable to connect to email server.
    """
```

- Do NOT add module-level docstrings (a `"""..."""` at the top of a file, before the imports). If a file needs framing that doesn't belong on a specific class/function, use a regular `#` comment instead.
- Types go in function signatures, NOT in docstrings.
  - If a default is present, DO NOT repeat it in the docstring unless there is post-processing or it is set conditionally.
- Focus on "why" rather than "what" in descriptions.
- Document all parameters, return values, and exceptions.
- Keep descriptions concise but clear.
- Ensure American English spelling (e.g., "behavior", not "behaviour").
- Do NOT use Sphinx-style double backtick formatting (` ``code`` `). Use single backticks (`` `code` ``) for inline code references in docstrings and comments.
