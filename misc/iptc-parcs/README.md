# iptc-parcs

Case study: three ways of letting an LLM drive the live [PARCS](https://github.com/alexeybogusevich/parcs8) cluster, compared on one Monte Carlo Value-at-Risk task.

| Strategy | How the model uses the PARCS tools |
|---|---|
| `baseline` | Native tool calling: one tool call per step, results sent back as tool messages (`iptc_parcs/baseline.py`) |
| `ptc` | Writes code in one `eval` tool call, executed after the call has fully streamed (`jitgen_openai.PtcAgent`) |
| `iptc` | Same, but each statement runs as soon as it's complete, while the call still streams, and the stream stops at the first error (`jitgen_openai.IptcAgent`) |

`ptc` and `iptc` each run in two languages, so every cell has five **arms**: `baseline`, `ptc-python`, `ptc-javascript`, `iptc-python` and `iptc-javascript`. Python runs in-process; JavaScript runs in QuickJS through `jitgen-js`, and calls each tool with one object of named arguments.

**Hypotheses:**
- **H1:** IPTC has the lowest total time.
- **H2:** IPTC uses fewer tokens while self-correcting, because it stops the model's stream at the first error.

## What is held identical

- **The system prompt** (`iptc_parcs/prompts/system.md`) and the task text, with no mention of a language or `eval`.
- **Model settings:** seed, temperature, the first turn forced to call a tool, the iteration limit, the time limit.
- **One tool surface** (`ExperimentBridge`, `iptc_parcs/bridge.py`), which every arm's tool calls pass through, including fault injection and recording. The baseline's native tool schemas use the same optional-parameter rule as the `eval` tool's signatures.
- **The only difference is how tools are exposed:** native tools for `baseline`, versus one `eval` tool whose description adds the code guidance for its language: `iptc_parcs/prompts/eval_code.md` (Python) or `eval_code_js.md` (JavaScript). For one language, that description is byte-identical for `ptc` and `iptc`.
- **One measurement layer for every arm:** an instrumented OpenAI client (`iptc_parcs/instrumentation.py`) records every model call, so no agent is measured differently.
- **One cap on tool results:** that same client truncates every tool message sent to the model at `--tool-result-limit` characters (20,000 by default): native tool replies for `baseline`, `eval` output for `ptc`/`iptc`. The system prompt states the limit. Without it, a baseline run once put all 2,000,000 Monte Carlo losses into the conversation and overflowed the model's context.

## The task

An equally weighted portfolio of 20 assets, with jointly normal returns and covariance Σᵢⱼ = sᵢsⱼ·0.5^|i−j|, where sᵢ = 0.01 + 0.001·i. The loss has an exact normal distribution, so VaR₉₉ = 0.017601 and CVaR₉₉ = 0.020165 are known analytically (`iptc_parcs/task.py`). The model has to run 2,000,000 Monte Carlo scenarios on the cluster; an answer counts as correct if both numbers are within `--tolerance` (1%).

## Scenarios

- **`natural`:** only the errors the model makes on its own.
- **`fault_compile`:** the first `create_session` of every run returns a transient compile error ("compilation service interrupted; resubmit the same source"), at the same step for all arms. This is the controlled self-correction case for H2.

## Live PARCS

Every run talks to the live cluster at `PARCS_SERVER_URL` (set it in `.env`, or pass `--parcs-url`), and live conditions are part of what is measured:
- **Cold starts:** the first layer after the cluster has scaled down waits minutes for worker pods (160-290s measured). Pods then stay warm across runs, so mostly the first runs of a session pay it. Arm order within a cell is a seeded shuffle recorded in `order`, so no arm always goes first. Compare arms on `cluster_seconds` and `order` as well as `wall_seconds`.
- **Flaky connections:** a dropped SSE connection fails its tool call with `ConnectionError`. A run with one that didn't produce a correct answer is an `infra_error` and is retried.
- **Languages aren't symmetric for H2:** a runtime error closes the stream early in both languages, but a JavaScript *syntax* error only surfaces once the whole `eval` call has streamed (see `libs/jitgen-js/README.md`), so there `iptc-javascript` behaves like `ptc`. Keep this in mind when comparing `early_closed_turns` or `est_output_tokens_failed_turns` across languages.
- **Timeouts:** `--executor-timeout` bounds one Python statement. In JavaScript it bounds only QuickJS's own running time, not time spent waiting on a tool, so only `--time-limit` cuts off a hung tool call there.

The unit tests never contact the cluster or a model. They run every arm, in both languages, through the real agents and executors against a fake PARCS, and check the tool schemas against `tests/unit_tests/fixtures/parcs_tools.json`, the live server's recorded tool list. Refresh it with `uv run scripts/snapshot_live_tools.py`.

## Running

```bash
uv sync --group test
cp .env.example .env    # OpenRouter key, LangSmith, PARCS URL

uv run pytest                                              # offline; both languages
uv run iptc-parcs --dry-run                                # list the planned runs
uv run iptc-parcs --out results/main                       # 5 reps x 5 arms x 2 scenarios
uv run iptc-parcs --languages javascript --out results/js  # baseline + JavaScript arms
uv run iptc-parcs --strategies iptc --scenarios natural --reps 1 --out results/smoke
uv run iptc-parcs --models lmstudio:qwen/qwen3.6-27b,openrouter:<id> ...
```

**Defaults:** 15 minutes and 12 iterations per run (`--time-limit`, `--max-iterations`), temperature 0.6, seeds `--seed-base` + rep, and local models loaded with a 65,536-token context (`--context-length`; LM Studio's own default of 8,192 is too small for a reasoning model).

**Protocol:**
- Every arm in a cell uses the same seed, and arm order is a seeded shuffle recorded in `order`.
- LM Studio reloads the model before every run, so no run gets a prompt-cache head start; the reload isn't timed.
- A lock file stops two experiments from sharing the model server.
- Runs already in `runs.csv` are skipped, so an interrupted experiment resumes.
- Runs that fail for infrastructure reasons (`infra_error`) are retried up to `--max-infra-retries` times.

## Output (CSV, in `--out`)

**`runs.csv`**, one row per run:

| Column | Meaning |
|---|---|
| `model`, `scenario`, `strategy`, `language`, `rep` | The run's identity; `language` is empty for `baseline` |
| `status` | `answered_correct`, `answered_wrong`, `no_answer`, `output_truncated` (the last model call hit the length limit), `context_overflow` (a request exceeded the model's context), `iteration_limit`, `time_limit`, `agent_error`, or `infra_error` (tools or model server failed, and the run didn't produce a correct answer) |
| `wall_seconds` | Total run time |
| `iterations` | Model round trips (agent-loop iterations) |
| `failed_iterations`, `self_correction_iterations`, `successful_iterations` | Iterations whose code or tool calls failed; tool-calling iterations after the first failure; tool-calling iterations without failure |
| `early_closed_turns` | Model streams the agent closed before they finished (IPTC early exit) |
| `truncated_tool_results` | Tool messages cut to `--tool-result-limit` in the run's last request |
| `est_*_tokens` | Input, output and reasoning tokens counted with tiktoken `o200k_base`: the same proxy for every arm, and available even for streams closed early. Not the model's own tokenizer |
| `usage_*_tokens` | Provider-reported usage, where the stream reached its final usage chunk |
| `native_*_tokens` | OpenRouter's server-side counts per generation, which include streams closed early |
| `est_output_tokens_failed_turns`, `est_tokens_after_first_failure` | H2 metrics: output spent in failing iterations, and all tokens after the first failure |
| `model_seconds`, `tool_seconds`, `overlap_seconds` | Time the model was streaming, time tools were running, and how much of the two overlapped (IPTC's gain) |
| `cluster_seconds` | Sum of `run_layer` `totalElapsedSeconds`: server-side cluster time |
| `seconds_per_iteration`, `est_tokens_per_iteration` | Derived per-iteration figures |

**`turns.csv`** has one row per model call: timings, including time to first chunk and first reasoning, content and arguments; `completed` or closed early; `finish_reason`; chunk counts; token estimates and usage; and whether the iteration failed.

**`tool_calls.csv`** has one row per PARCS tool call: timing, success, whether it was injected, and `totalElapsedSeconds`.

## Checking that early exit really saves tokens

Output tokens not received by the client only count as saved if the server stops generating when the stream is closed. `scripts/check_early_close.py` compares the time to first token of a request sent right after closing a long generation against a baseline. It reloads the model with a single request slot first; with several slots, a continued generation would just take another slot, and the test would show nothing. If it reports that the server keeps generating, report H2 for this server as client-received tokens only. For OpenRouter, the `native_*` columns show what was actually generated.

**Result on LM Studio with qwen/qwen3.6-27b (2026-09-27):** the server stops on close. The time to first token right after a close was 0.38s, against 0.34s at baseline, where a continued generation would have added about 160s.
