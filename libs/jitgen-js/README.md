# jitgen-js

JavaScript guest-language support for [JitGen](../jitgen): an ANTLR4-based
`StatementExtractor` (`JavaScriptAntlrExtractor`) and a QuickJS-based
`BaseExecutor` (`QuickJsExecutor`), wired together by
`create_javascript_session()`.

```python
from jitgen import StreamDriver
from jitgen.segmenters import markdown_code
from jitgen_js import create_javascript_session

session = create_javascript_session()
driver = StreamDriver(session, markdown_code("javascript"))

for chunk in model_stream:  # however your LLM client streams text
    for output in await driver.apush(chunk):
        print(output, end="")
for output in await driver.afinish():
    print(output, end="")
```

## Known deviations from post-hoc (whole-script) execution

JitGen dispatches one top-level statement at a time rather than running the
whole generated script as a single unit. For JavaScript this has real,
observable consequences beyond the Python port's:

- **Top-level `const` can be redeclared across statements.** `QuickJsExecutor`
  rewrites top-level `const` to `var` (`SourceTransform.
  TOP_LEVEL_CONST_TO_VAR`) so that a `const` declared in one dispatched
  statement doesn't collide with a `const` of the same name in a later one —
  necessary because each statement runs as its own `eval` call, not as
  continuous script text. Whole-program execution of the same source would
  throw on a genuine `const` redeclaration; incremental dispatch will not.
- **Function hoisting across statement boundaries does not work.** A
  statement that calls a function declared later in the same buffer fails
  under incremental dispatch even though it would succeed if the whole
  script ran at once. Python has no hoisting to lose, so this is a
  JS-specific gap the Python port's design doesn't have to contend with.
- **JS syntax errors are only ever raised at end-of-stream flush, never
  mid-stream** — a deviation from the method's headline benefit (surfacing
  errors "as early as the offending prefix is complete"). ANTLR4's ALL(*)
  parsing algorithm lacks the "valid-prefix property" the method's
  Admissible-parser precondition assumes: when a buffer is genuinely
  incomplete (an unclosed `(`/`{`/string), ALL(*)'s unbounded lookahead can
  report a failure token arbitrarily far from the actual truncation point,
  not at it — unlike the Python/Lark extractor's LALR parser, whose lexer
  fails exactly where a chunk boundary lands. `JavaScriptAntlrExtractor`'s
  disambiguation policy (see `_parse_or_recover`'s docstring) is therefore
  to treat every mid-stream parse failure as "more input needed" and defer
  classification entirely to the final flush. A consequence: `Session.
  push`'s `has_error` flag, which callers poll to abort the model stream
  early, will not trip for a JS syntax error the way it does for Python's.
- **`QuickJsExecutor.acancel()` is a no-op.** `quickjs_rs` (confirmed by
  inspecting its actual API, not assumed) exposes no primitive to interrupt
  a call already in progress — cancelling the Python-side future only
  abandons the caller's own await while the worker thread keeps running the
  JS engine until that call's own `timeout` elapses regardless, since
  QuickJS execution blocks the worker thread's event loop synchronously for
  the call's whole duration. The only real bound on a runaway statement is
  `timeout` itself.

## Development

```bash
uv sync --group test
uv run pytest
uvx ruff format .
uvx ruff check --fix .
```

The generated ANTLR lexer/parser under `jitgen_js/_grammar/` is vendored,
not built at install time — see `scripts/regenerate_grammar.sh` if the
grammar ever needs to be regenerated (requires a Java runtime and the
ANTLR 4.13.2 tool jar; not required to just install/use this package).
