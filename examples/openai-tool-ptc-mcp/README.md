# openai-tool-ptc-mcp

Incremental Programmatic Tool Calling (**IPTC**) with real
[Model Context Protocol](https://modelcontextprotocol.io) tools.

As in `../openai-tool-ptc`, the model gets one tool, `eval`, whose `code`
argument is Python source (or JavaScript, see
[JavaScript](#javascript)), and `jitgen` executes each top-level statement of
`code` as soon as it is complete, while the tool call is still streaming.
The difference: the helpers the generated code calls are not mocks but real
MCP tools, so a tool's network latency overlaps with the model generating
the rest of the code.

Both scripts are thin wiring around two reusable `jitgen_openai` classes:

- **`McpToolBridge`** connects to one MCP server (`.stdio(...)`,
  `.streamable_http(url)` or `.sse(url)`) and exposes each of its tools as an
  `async def name(**kwargs) -> str` callable for the executor.
- **`IptcAgent`** runs the multi-turn loop: stream a turn, execute its `eval`
  call incrementally, feed the output back, and repeat until the model
  answers in plain text. Variables persist across turns.
- **`PtcAgent`** is a demo-only baseline for comparison: the same loop,
  executor, tools and tracing, but it waits for the whole `eval` call before
  running its code in one go, as conventional PTC does.

## Scripts

- **`agent_example.py`**: fetches two Wikipedia articles through the
  official read-only
  [`mcp-server-fetch`](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch)
  server (launched over stdio via `uvx`, no API key) and compares their
  length. The first `fetch` runs while the model is still writing the
  second. The prompt asks for exactly one `eval` call that fetches both pages
  and prints the comparison: without that, models tend to fetch one page
  per call, or to compare in a second call, paying a model round trip each.
- **`parcs_example.py`**: runs a parallel C# job on the
  [PARCS](https://github.com/alexeybogusevich/parcs8) cluster. See
  [PARCS](#parcs) below.
- **`compare_example.py`**: runs the fetch task through `PtcAgent`, then
  `IptcAgent`, and prints both wall times. Both send the same model `seed`
  (`COMPARE_SEED`, default 42), so they normally write the same code and
  differ only in when it runs. A seed makes sampling repeatable only if the
  server honors it (LM Studio does), so compare over several seeds.

Every script takes `--language python` (the default) or
`--language javascript`, which picks the matching system prompt and executor.

`utils.py` holds the shared client, progress printing and agent setup.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (provides `uvx`, which launches
  `mcp-server-fetch` on demand).
- [LM Studio](https://lmstudio.ai) with its local server running (default
  `http://localhost:1234`) and a tool-calling model loaded. Any other
  OpenAI-compatible endpoint works too: set `OPENAI_BASE_URL` and
  `OPENAI_API_KEY`.
- Internet access (Wikipedia for `fetch`, the PARCS endpoint for PARCS).

```bash
export OPENAI_MODEL="your-loaded-model-id"
uv run agent_example.py
uv run agent_example.py --language javascript
uv run parcs_example.py   # see PARCS below for options
uv run compare_example.py # PtcAgent vs IptcAgent on the fetch task
```

LangSmith tracing is opt-in: set `LANGCHAIN_TRACING=true` and
`LANGCHAIN_API_KEY`. Each `arun()` is one `IptcAgent` trace. Under it are the
model calls and one `eval` tool run per turn, and each MCP tool call made by
the executed code is a child of its `eval` run. Because code runs while it
streams, an `eval` run overlaps the model call that is generating it.
`PtcAgent` traces have the same shape, named `PtcAgent`, with each `eval` run
starting only after its model call ends. Every run carries a `ptc_strategy`
metadata tag (`incremental` or `buffered`) for filtering.

## What to expect

- The streaming caveats in `../openai-tool-ptc/README.md` apply unchanged:
  the benefit needs token-level streaming of the tool-call arguments, and a
  loop around the tool calls is one statement, so it only runs once fully
  streamed. The prompts forbid such loops for that reason.
- `mcp-server-fetch` warns once when Node.js is missing and falls back to a
  pure-Python extractor; it still works.
- When a statement fails, `IptcAgent` stops reading the model's stream
  immediately and closes the truncated `arguments` into valid JSON before
  sending them back in the history. LM Studio rejects truncated JSON there.

## JavaScript

With `--language javascript` (`IptcAgent(..., language="javascript")`), the
model writes JavaScript. [`jitgen-js`](../../libs/jitgen-js) parses it with
ANTLR and runs it in an embedded QuickJS engine, not Node.js, so there is no
`require`, `process` or `setTimeout`, and `console.log` is the only output.
JavaScript has no keyword arguments, so each MCP tool takes one object of
named arguments, `await fetch({url: "...", max_length: 200000})`, and the
`eval` tool describes the tools that way. The JavaScript prompts
(`JAVASCRIPT_SYSTEM_PROMPT` in `agent_example.py`, `parcs_prompt_js.md`)
follow the Python ones. The differences come from the
[`jitgen-js` deviations](../../libs/jitgen-js/README.md):

- The prompts require a semicolon after every statement. The extractor
  releases a statement at its line end even without one, so a next line
  that JavaScript would join onto it (one starting with `(`, `[`, `.` or a
  backtick) would be split off and fail instead.
- They forbid functions of your own, because a declaration is not hoisted
  across statements that run one at a time.
- The PARCS prompt puts the C# source in `` String.raw`...` ``, so the C#
  must not contain a backtick or `${`.
- A syntax error only surfaces once the whole `eval` call has streamed, not
  mid-stream as in Python.
- `executor_timeout` only bounds QuickJS's own running time, not time spent
  awaiting an MCP tool, so a hung tool call is not cut off.

## PARCS

[PARCS](https://github.com/alexeybogusevich/parcs8) runs C# code across
worker pods and exposes it over MCP (legacy HTTP+SSE). The endpoint is not
kept in the repository: set `PARCS_SERVER_URL` to the cluster's MCP SSE
endpoint (e.g. `http://<host>:8080/sse`) in `.env`. The cluster's web UI shows
sessions and layers as they run.

A job is a sequence of **layers**. The model writes one flat `eval` call
that calls `get_cluster_info` to size the job, compiles the body of a C#
`ExecuteAsync` method with `create_session`, runs a parallel `run_layer`,
and chains a single-worker aggregation layer through `previousLayerId`.
Each statement runs as soon as it is written, so compilation and the
parallel layer run while the model is still writing the rest.
`parcs_prompt.md` holds the system prompt: the working rules and the tool
contract as the live server implements it. `parcs_prompt_js.md` is the same
prompt for `--language javascript`.

```bash
uv run parcs_example.py                    # default task: count primes, IptcAgent
uv run parcs_example.py --agent ptc        # same task with PtcAgent
uv run parcs_example.py --seed 42          # reproducible model sampling
uv run parcs_example.py --task var         # Monte Carlo VaR/CVaR preset
uv run parcs_example.py --language javascript  # JavaScript eval code
uv run parcs_example.py --task "..."       # any task text
```

The default `primes` task counts the primes in [1,000,000,000,
1,010,000,000). The expected answer, 482,449, was computed independently,
and the script reports whether the agent's answer contains it. The `var`
preset is adapted from the
[PARCS-Agent benchmark](https://github.com/alexeybogusevich/parcs8/tree/master/misc/mcp_eval),
minus its numpy-specific seeding, so it has no fixed expected answer.

Things to know:

- The first layer can take several minutes while the cluster starts worker
  pods; later layers reuse them. The script allows 600s per statement and
  16 model turns, for retries after compile errors.
- `run_layer` returns camelCase JSON, but the stored layer result that the
  next layer receives in `PreviousLayerResultJson` (and that
  `get_layer_result` returns) uses PascalCase names such as `Results` and
  `OutputData`. The prompt spells this out, because the wrong casing makes
  the aggregation layer fail with "The given key was not present in the
  dictionary".
- The server marks every tool parameter as required but documents
  defaults; `McpToolBridge` shows those parameters as optional, and the
  server accepts calls that omit them.
- The SSE connection is occasionally flaky (`Connection closed`,
  `ReadTimeout`). If `run_layer` returns status `Running`, the layer kept
  going and `get_layer_result` fetches it later.

## Using other MCP servers

Pass any bridge, or several; `IptcAgent` merges their tools and
descriptions:

```python
extra = McpToolBridge.stdio(
    StdioServerParameters(command="uvx", args=["some-other-mcp-server"])
)
agent = IptcAgent(client, model, bridges=[fetch_bridge, extra], ...)
```
