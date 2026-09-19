# openai-tool-ptc

Programmatic Tool Calling (PTC) example: a model is given one tool, `eval`,
whose `code` argument is source code — the [LangChain deepagents
interpreter](https://docs.langchain.com/oss/python/deepagents/interpreters)
convention. Instead of waiting for the tool call's `arguments` to finish
streaming before running anything, `jitgen` (via `jitgen-openai`'s
`OpenAIToolCallSegmenter`) executes each top-level statement of `code` as
soon as its boundary is provably fixed, while the rest of the JSON object is
still arriving. The captured output is then sent back to the model as the
tool result so its final answer comes from the real, computed value.

Two scripts, same idea in two guest languages:

- **`python_example.py`** — `code` is Python, executed via `jitgen`'s
  `create_python_session()`/`InProcPythonExecutor`.
- **`js_example.py`** — `code` is JavaScript, executed via `jitgen-js`'s
  `create_javascript_session()`/`QuickJsExecutor`. See
  `libs/jitgen-js/README.md` for JS-specific deviations from whole-script
  execution (e.g. mid-stream syntax errors are only ever surfaced at flush,
  not as-you-type, unlike the Python/Lark extractor).

On top of that, the executed `code` in both scripts has access to a second,
inner tool — `lookup_price`/`lookupPrice`, a mock price lookup injected
into the executor's namespace via `InProcPythonExecutor(tools=...)` /
`QuickJsExecutor(tools=...)`. This is the other half of PTC — the
*generated code* reaching out to a tool mid-computation, not just the model
making a tool call. `_lookup_price` is a genuine `async def` — no
synchronous bridge — and the LLM-generated code is instructed to (and, in
practice, does) call it as `await lookup_price(...)` / `await
lookupPrice(...)`, a real top-level `await` directly in a flat sequence of
statements, not wrapped in an `async def main()`. `QuickJsExecutor` needs
no changes to support this: `quickjs_rs`'s `Context.register` auto-detects
an `async def` and exposes it to JS as a function returning a `Promise`.
`InProcPythonExecutor` does need one: it now compiles every dispatched
statement with `ast.PyCF_ALLOW_TOP_LEVEL_AWAIT` and runs it via `eval()`
rather than `exec()`, driving the resulting coroutine to completion with
`asyncio.run()` when a statement's `co_flags` show it contains a top-level
`await` (see `libs/jitgen/jitgen/executors/python.py`). The `await`ed
lookup blocks (asynchronously) for `_LOOKUP_DELAY_SECONDS` — a stand-in
for real network latency — so a task that calls it once per symbol lets
you see jitgen dispatch and execute each lookup/log-output pair to the
background worker while the model is still streaming the statements that
come after them — code execution genuinely overlapping with generation,
not just with tool-call JSON parsing.

## Prerequisites

- [LM Studio](https://lmstudio.ai) running locally with its local server
  enabled (`Developer` tab → `Start Server`, default `http://localhost:1234`).
- A tool-calling-capable model loaded in LM Studio. Set `OPENAI_MODEL` to its
  LM Studio identifier (see the model list at `GET /v1/models`, or the
  identifier shown in LM Studio's UI) — there's no universal default, since
  it depends entirely on what you have loaded.

```bash
export OPENAI_MODEL="your-loaded-model-id"
uv run python_example.py   # or: uv run js_example.py
```

Any other OpenAI-compatible endpoint works too — set `OPENAI_BASE_URL` and
`OPENAI_API_KEY` to match.

LangSmith tracing is wired in the same way as `misc/mbpp`'s providers (the
client is wrapped with `langsmith.wrappers.wrap_openai`) and is opt-in: set
`LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to send a trace of both
calls; otherwise the wrapper is a no-op passthrough.

## What to expect — and a real caveat

Whether the tool call's `code` argument actually streams in incrementally
(proving jitgen's time-to-first-output benefit) or arrives all at once in a
single delta (the example still runs correctly either way, it just won't
visibly demonstrate the benefit) depends entirely on the model/runtime.
Both scripts count the argument deltas they receive and print a warning if
only one arrived.

Two concrete compatibility notes from getting this running against LM
Studio's llama.cpp-based server:

- **`tool_choice`**: the object form (`{"type": "function", "function":
  {"name": "eval"}}`) to *force* the `eval` tool was rejected
  (`Invalid tool_choice type: 'object'`) — only the string values
  (`"auto"`/`"required"`/`"none"`) are supported. Both scripts use
  `tool_choice="required"`, which is just as deterministic given only one
  tool is offered.
- **Streaming granularity**: with `qwen/qwen3.6-27b` loaded locally, the
  `code` argument arrived across multiple deltas (no buffering warning) for
  both scripts, and each run produced the correct answer — five mock
  prices looked up via `lookup_price`/`lookupPrice` and a correct total
  cost — end to end, including the round trip back through a second
  streamed call for the final natural-language answer. In the JavaScript
  run, the model wrote its lookups as `const` declarations; `QuickJsExecutor`
  silently rewrites top-level `const` to `var` (see
  `libs/jitgen-js/README.md`), so this worked without any special handling.

If your loaded model doesn't support tool calling at all, the first
streamed call will simply never produce a `tool_calls` delta, and `run()`
returns `"(model did not call the eval tool)"`.
