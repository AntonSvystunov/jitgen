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

Three scripts:

- **`python_example.py`** — `code` is Python, executed via `jitgen`'s
  `create_python_session()`/`InProcPythonExecutor`.
- **`js_example.py`** — `code` is JavaScript, executed via `jitgen-js`'s
  `create_javascript_session()`/`QuickJsExecutor`. See
  `libs/jitgen-js/README.md` for JS-specific deviations from whole-script
  execution (e.g. mid-stream syntax errors are only ever surfaced at flush,
  not as-you-type, unlike the Python/Lark extractor).
- **`benchmark.py`** — times jitgen's incremental dispatch against
  "regular" (buffer, then execute) PTC on identical, recorded model output.
  See [Benchmark](#benchmark) below.

Shared code (the mock `lookup_price` tool, the OpenAI client, and the
stream-execute-answer loop) lives in `utils.py`.

On top of that, the executed `code` in both scripts has access to a second,
inner tool — `lookup_price`/`lookupPrice`, a mock price lookup injected
into the executor's namespace via `InProcPythonExecutor(tools=...)` /
`QuickJsExecutor(tools=...)`. This is the other half of PTC — the
*generated code* reaching out to a tool mid-computation, not just the model
making a tool call. `utils.lookup_price` is a genuine `async def` — no
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
lookup blocks (asynchronously) for `LOOKUP_DELAY_SECONDS` — a stand-in
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
Both scripts count the non-empty argument deltas they receive and print a
warning if only one arrived.

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
streamed call will simply never produce a `tool_calls` delta, and `utils.run()`
returns `"(model did not call the eval tool)"`.

Two more findings, from building `benchmark.py` (below), worth knowing
before you draw conclusions from either script's timing:

- **Empty deltas don't count as streaming.** `qwen/qwen3.6-27b`
  sometimes sends the whole `arguments` JSON as a *single* non-empty delta
  preceded by one empty id/name-only delta, which is why the buffering
  warning counts only non-empty deltas. `benchmark.py` prints the recorded
  delta count and total generation time directly, so you can see the real
  granularity.
- **A `for`/`while` loop around the lookups defeats the benefit even with
  genuine token-level streaming.** jitgen dispatches at top-level-*statement*
  granularity — a loop is one statement, so its whole body (all five
  lookups) only runs once the loop's source is fully streamed and its
  boundary closes, which is functionally the same as the buffer-then-execute
  baseline. `gpt-oss-20b` wrote exactly this on an earlier, weaker version
  of the prompt; both `SYSTEM_PROMPT`s now explicitly forbid loops/
  comprehensions/`Promise.all`/`asyncio.gather` over the symbols for this
  reason.

## Benchmark

`benchmark.py` measures the thing this whole example exists to demonstrate:
does executing code *while it's still streaming in* actually save time over
the conventional "wait for the full tool call, then run it" pattern most
PTC/`eval`-tool integrations use?

It benchmarks both guest languages: Python (`InProcPythonExecutor`) and
JavaScript (`QuickJsExecutor`). To keep the comparison fair, it makes
exactly **one** real streamed `eval` tool call per language and records the exact arrival timing of every `arguments` delta.
Both strategies then *replay* that identical, recorded cadence — several
times each, for averaging — so a difference in results can only come from
*when* each strategy starts executing, never from the model writing
different code on a second call, or from ordinary network jitter:

- **jitgen (incremental PTC)** — the same `StreamDriver`/
  `OpenAIToolCallSegmenter` pipeline `python_example.py`/`js_example.py`
  use; each delta is pushed in as it "arrives".
- **regular PTC (buffer, then execute)** — accumulate every delta first,
  `json.loads` the complete `arguments`, then run the whole `code` in one
  `aexecute()` call on that language's executor. It reuses the same executor
  jitgen's own path uses (not a hand-rolled `exec()`), so the comparison
  isolates dispatch timing, not interpreter quality.

It reports **time to first output** (when the first piece of execution
output became available) and **total time** (when everything finished),
mean ± stdev over `BENCHMARK_TRIALS` replays (default 5), plus a speedup
figure and a correctness check that every trial of both strategies produced
byte-identical output, for each language.

```bash
export OPENAI_MODEL="your-loaded-model-id"
uv run benchmark.py          # both languages
uv run benchmark.py js       # or just one: python / js
```

**A real measured run** (Python only, from before the JavaScript comparison
was added), `openai/gpt-oss-20b` (128 argument deltas over
1.17s of genuine token-level generation, flat non-loop statements):

```
jitgen (incremental PTC):
  time to first output: mean 1.23s (± 0.18s)
  total time:            mean 2.76s (± 0.18s)

regular PTC (buffer, then execute):
  time to first output: mean 4.22s (± 0.00s)
  total time:            mean 4.22s (± 0.00s)

Speedup (total time): 1.53x (1.46s saved on average)
Outputs identical across every trial and strategy: True
```

Time to first output is ~3.4x faster and total time is 1.53x faster — the
five `lookup_price` calls (0.3s each, 1.5s sequential total, since they all
run on one worker thread either way) mostly overlap with the ~1.17s of
ongoing generation under jitgen, and are fully additive *after* generation
under the baseline. The achievable speedup is bounded by
`min(generation_time, total_tool_latency)`, so it depends on both the task
and the model: a model that streams `arguments` in one giant chunk (see
above), or writes the lookups inside a loop, will show little to no
difference — that's not a bug in the benchmark, it's an accurate report of
there being no incremental-dispatch opportunity to exploit under those
conditions.
