# JitGen: Grammar-Guided Incremental Execution of LLM Generated code

JitGen interleaves parsing and execution *with* LLM generation: instead of waiting for a model to finish streaming a code block before running it (generate-then-execute), JitGen parses the growing output buffer against the target language's grammar and dispatches each top-level statement to a persistent interpreter as soon as its boundary is provably fixed — i.e. as soon as no future token could still change it. This lowers time-to-first-output and surfaces syntax/runtime errors as early as the offending prefix is complete, without touching model decoding.

See [`docs/METHOD.md`](docs/METHOD.md) for the formal spec (principles, preconditions, and terminology used throughout the codebase).

## Repository layout

- [`libs/jitgen`](libs/jitgen) — the library. A small pipeline of independently-swappable pieces (segmenter, grammar-aware extractor, executor, session) wired together for Python via `jitgen.create_python_session()`. See [`libs/jitgen/README.md`](libs/jitgen/README.md) for the package overview and `CLAUDE.md` for an architecture walkthrough.
- [`docs/METHOD.md`](docs/METHOD.md) — the method specification.
- [`examples/openai-example`](examples/openai-example) — a runnable example that streams a coding agent's response through JitGen so each top-level statement executes as soon as it's written.
- [`misc/mbpp`](misc/mbpp) — evaluation of the JitGen algorithm on a modified MBPP dataset.

## Getting started

The library is developed from `libs/jitgen/` (its `pyproject.toml`, not a root one, is the package root):

```bash
cd libs/jitgen
uv sync --group test
uv run pytest
```

Requires Python >=3.13. See `CLAUDE.md` for full development, linting, and testing conventions.
