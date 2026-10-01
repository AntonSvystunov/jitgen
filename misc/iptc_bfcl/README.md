# iptc-bfcl

Case study: three ways of letting an LLM call functions, compared on the single-turn categories of the [Berkeley Function Calling Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (BFCL). The harness is adapted from [`misc/iptc-parcs`](../iptc-parcs/README.md); its strategies, arms and measurement layer are the same.

| Strategy | How the model uses the functions |
|---|---|
| `baseline` | Native tool calling. The tool calls of one turn run concurrently, and their results go back as tool messages (`iptc_bfcl/baseline.py`) |
| `ptc` | Writes code in one `eval` tool call, executed after the call has fully streamed (`jitgen_openai.PtcAgent`) |
| `iptc` | Same, but each statement runs as soon as it's complete, while the call still streams. The stream stops at the first error (`jitgen_openai.IptcAgent`) |

## Run modes

BFCL scores a single model response in its single-turn categories. The calls in that response are parsed and graded; no results go back to the model, and there is no closing text turn. By default this harness does the same.

**Single response (the default):** a run ends once the first response, the forced tool call, has streamed and its calls have finished executing.
- **What `wall_seconds` covers:** generation plus tool execution, with no closing turn diluting the difference between arms.
  - `iptc` runs the calls while the response streams.
  - `ptc` runs them after it has streamed.
  - `baseline` runs them concurrently after it has streamed.
- **Grading:** the run's calls all come from that one response, so `correct` and `first_turn_correct` agree.
- **Error recovery:** a failing call isn't retried, and a run whose response fails partway is graded on the calls that succeeded.

**`--full-cycle`:** runs the whole agent loop instead. Tool results go back to the model until it answers in text, or `--max-iterations` is reached. This mode measures self-correction and the full turnaround.

The mode is part of each run's key (`mode`: `single_response` or `full_cycle`), so both can share an `--out` directory.

`ptc` and `iptc` each run in Python (in-process) and JavaScript (QuickJS), so every cell has five **arms**: `baseline`, `ptc-python`, `ptc-javascript`, `iptc-python` and `iptc-javascript`.

## Simulated functions

BFCL ships function docs and accepted answers, but no implementations. Every function is therefore served by `BfclToolBridge` (`iptc_bfcl/tools.py`) as an async function. On each call it:

1. Binds positional and keyword arguments to the documented parameters.
2. Fails with a `TypeError` on an unknown or missing required parameter, as a real signature would.
3. Waits `--tool-delay` seconds.
4. Returns a deterministic echo: `{"arguments": ..., "function": ..., "note": ..., "result_id": ..., "status": "ok"}`.

Every call, failed ones included, is recorded under its BFCL name for grading.

The echo's `note`, and a rule in the system prompt, say that the functions are simulated and return no data. An early smoke run used a bare echo, and the models reacted in ways that skewed iterations and time between arms:
- Some reasoned until the time limit while waiting for the missing numbers.
- Code arms computed the requested values themselves in extra `eval` turns.

`--tool-delay` is a grid axis (`--tool-delay 0.1,1.0`), because IPTC's gain comes from overlapping tool latency with generation.

## What is held identical

- **The system prompt** (`iptc_bfcl/prompts/system.md`) mentions neither a language nor `eval`. When a BFCL entry has its own system message, it is appended for every arm.
- **Model settings:** seed, temperature, the first turn forced to call a tool, the iteration limit, the time limit.
- **One tool surface per entry** (`BfclToolBridge`), which every arm's calls pass through.
- **One set of names:**
  - Dotted BFCL names (`math.factorial`) become `math_factorial` in every arm, since OpenAI function names and code identifiers can't contain a dot.
  - BFCL's non-JSON-Schema types (`dict`, `float`, `tuple`, `any`) are converted once, for every arm.
- **The same parameter docs:**
  - The baseline gets each function as a native JSON schema.
  - The code arms get the same information as text in the `eval` tool's description: signature, type, required or optional, enum, default and description of every parameter (nested fields included).
  - That text is followed by the code guidance for the language (`prompts/eval_code.md`, `eval_code_js.md`), and is byte-identical for `ptc` and `iptc`.
- **One measurement layer and one cap on tool results** (`iptc_bfcl/instrumentation.py`), as in iptc-parcs.

### Concurrency is not symmetric, by design

- The **baseline** runs the calls of one turn concurrently. A model emits several calls at once only when they're independent, so this is the fairest native baseline for BFCL's `parallel` categories.
- The **code arms** are told to write one `await` per statement, with no `asyncio.gather` or `Promise.all`, as in iptc-parcs.
- As a result, `ptc` pays the delays one after another, after generation. `iptc` pays them one after another too, but overlapped with generation.

This trade-off is part of what's measured.

## Categories and grading

The categories are `simple`, `multiple`, `parallel`, `parallel_multiple`, and their `live_*` versions: 2,351 entries at the pinned revision. `irrelevance` is left out, because every arm forces a tool call on its first turn. Multi-turn categories are out of scope.

Calls are graded by BFCL's AST checker, vendored in `iptc_bfcl/grading.py`:
- **The checker:** the Python-language path; Apache-2.0, attribution in the file header.
- **Category rules:**
  - `simple`: exactly one call.
  - `multiple`: one call, to the right function.
  - `parallel*`: every expected call matched once, in any order.
- **Value rules, as in BFCL:**
  - an optional parameter may be omitted when its accepted values include `""`;
  - strings are compared case-insensitively, ignoring spaces and `,./-_*^`;
  - an int is accepted where a float is expected.

  The int-for-float rule also covers JavaScript, which has no int/float distinction: `7.0` reaches Python as `7`.

Two grades are recorded:
- **`correct`, the primary metric:** every *successful* call of the run (in a single-response run, that is one response), with exact repeats counted once. Calls spread over several turns still count, and a retry that repeats a call which already succeeded doesn't spoil the match.
- **`first_turn_correct`, closest to BFCL's own scoring of one response:** only the successful calls of the first turn that called a tool. A first response that failed outright (invalid code or JSON, every call rejected) scores false, even when a retry got it right.

`correct` judges the calls, whatever ended the run. `status` is:
- `answered_correct` or `answered_wrong` when the run ended as its mode intends;
- otherwise why it didn't: `output_truncated`, `context_overflow`, `iteration_limit`, `time_limit`, `agent_error` or `infra_error`.

**Known dataset quirks:**
- `live_multiple_1052-79-0`'s answer is stored under the id `live_multiple_1052-279-0`. It is matched by line instead.
- `simple_363`'s answer names `find_closest` for `restaurant_search.find_closest`. Like BFCL, `simple` is checked against its single function, whatever name the answer uses.
- About 9 entries can't be answered correctly. Their answers use a parameter missing from the function's schema, or leave a required parameter optional. Feeding every entry's own ground truth through the simulated functions and the checker grades the other 2,342 entries correct.

## Running

The dataset is downloaded once, at a pinned revision, to `~/.cache/iptc-bfcl/<revision>/` (`--data-dir` to change).

```bash
uv sync --group test
cp .env.example .env    # OpenRouter key, LangSmith

uv run pytest                                                  # offline; every arm, both languages
uv run iptc-bfcl --dry-run                                     # list the planned runs
uv run iptc-bfcl --out results/main                            # 20 entries per category x 5 arms
uv run iptc-bfcl --limit 0 --tool-delay 0.1,1.0 --out results/full
uv run iptc-bfcl --full-cycle --out results/main               # whole agent loop, same out dir
uv run iptc-bfcl --categories parallel,parallel_multiple --ids parallel_0,parallel_multiple_1 ...
uv run iptc-bfcl --models lmstudio:qwen/qwen3.6-27b,openrouter:<id> ...
```

**Defaults:**
- **Sample:** 20 entries per category (`--limit`, 0 for all), seeded by `--sample-seed`.
- **Grid:** 1 rep, and a 0.1 s tool delay.
- **Per run:** a single response (`--full-cycle` for the agent loop, with 4 iterations by default, `--max-iterations`), 3 minutes (`--time-limit`), and 30 s per executed statement (`--executor-timeout`).
- **Model:** temperature 0.6, and seeds `--seed-base` + rep.
- **Context:** local models are loaded with a 65,536-token context.

**Protocol:**
- **Within a cell:** cells are (model, reasoning effort, tool delay, rep, entry). Every arm in a cell uses the same seed, and arm order is a seeded shuffle recorded in `order`.
- **Model reloads:** LM Studio reloads the model once, before its first run. `--reload-each-run` restores iptc-parcs' reload before every run, so no run gets a prompt-cache head start. That is slow at BFCL's scale.
- **Run lock:** the lock file is the one iptc-parcs uses, so the two experiments never share a model server.
- **Resuming:** runs already in `runs.csv` are skipped, and `infra_error` runs are retried up to `--max-infra-retries` times.

The OpenRouter-only options `--reasoning-effort` and `--provider` work as in iptc-parcs.

## Report: paper tables and figures

```bash
uv sync --group test --group analysis
uv run iptc-bfcl-report results/check-multicall --exclude gemini   # writes results/check-multicall/report/
```

`--exclude` drops models whose id contains the given text (for example a throttled, incomplete model), and `--no-figures` writes only the tables, which need no extra dependency.

**Tables** are written to `report/tables/` as Markdown, LaTeX (booktabs, with `\label{tab:<name>}`) and CSV. `report/tables.md` collects all of them.

| Table | Content |
|---|---|
| `accuracy` | Correct runs per model and arm, with 95% Wilson intervals |
| `latency` | Median wall, generation, tool and overlap time, and wall − generation, per model and arm |
| `paired` | IPTC against PTC, paired by entry: median difference with a 95% bootstrap CI and an exact sign test, for both wall time and wall − generation |
| `failures` | Wrong arms per entry and model |

**Figures** are written to `report/figures/` as PDF (vector, for the paper) and PNG.

| Figure | Content |
|---|---|
| `timeline_example` | One response with identical generations in both arms, run by PTC and by IPTC on one time axis: when the code was written, and when each tool call ran |
| `paired_savings` | Per-entry PTC − IPTC differences with medians and 95% CIs, for wall time and for wall − generation |
| `overlap_mechanism` | The share of tool time IPTC hid in generation, against how long the model took to write one call; the tool latency is marked |
| `time_breakdown` | Mean run time split into generation, generation overlapped by tool calls, and the time after generation |
| `accuracy` | Accuracy per arm and model with Wilson intervals |

**Why wall − generation is reported:** IPTC changes when code runs, not what the model generates. When the two arms' generations differ, which happens whenever the provider isn't repeatable under a fixed seed, that variation swamps wall time. Wall − generation removes it. The `Identical output` column of `paired` shows how many pairs had exactly the same generation.

## Output (CSV, in `--out`)

**`runs.csv`**, one row per run:

| Column | Meaning |
|---|---|
| `model`, `category`, `entry_id`, `strategy`, `language`, `reasoning_effort`, `upstream`, `tool_delay`, `mode`, `rep` | The run's identity |
| `status`, `correct`, `first_turn_correct`, `grade_reason` | The grade (see above); `grade_reason` is the checker's error for a wrong run |
| `expected_calls`, `successful_calls` | Calls in the accepted answer; distinct successful calls made |
| `wall_seconds`, `iterations`, ... | The iptc-parcs metrics, minus the PARCS-specific `cluster_seconds` and `injected_faults` |

**`turns.csv`** has one row per model call, as in iptc-parcs. **`tool_calls.csv`** has one row per function call: BFCL name, `arguments` (JSON), timing, success, error and turn.

**Chunk times are arrival times.**
- **Why it matters:** when the `code` argument closes, `StreamDriver` waits for its last statements to finish before it reads the next chunks. Timestamping a chunk when the agent reads it would stretch IPTC's `last_token`, and with it `model_seconds` and `overlap_seconds`, over that execution.
- **What this harness does:** a reader task in the instrumented stream stamps each chunk on arrival instead. Text and token estimates are still counted as the agent consumes chunks, so `est_output` and `early_closed_turns` mean what they mean in iptc-parcs.
- **iptc-parcs:** it still stamps chunks on read, so its IPTC `model_seconds` and `overlap_seconds` are inflated by this effect.
