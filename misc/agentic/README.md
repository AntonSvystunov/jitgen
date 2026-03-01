# Agentic Evaluation – CodeAct Agent with LangGraph

Compares the performance of an AI agent (using the `xingyaow/codeact-agent-mistral` model) on the [MBPP-JITGen validation dataset](https://huggingface.co/datasets/AntonSvystunov/mbpp-jitgen-validation) with two execution strategies:

| Strategy | Description |
|----------|-------------|
| **sync** | Code extracted from `<execute>` tags is executed in one shot via `InProcPythonExecutor` |
| **async** | Code extracted from `<execute>` tags is fed line-by-line to **JITGen** for incremental parsing + execution |

Both strategies use a multi-turn **CodeAct agent loop** (up to 3 turns by default) implemented as a LangGraph `StateGraph`. The agent generates code, receives execution output, and can self-correct across turns – matching the approach described in the [CodeAct paper](https://arxiv.org/abs/2402.01030).

## Architecture

```
START → agent → [has <execute>?]
                  ├─ yes → execute_code → [turns exhausted / timeout?]
                  │                         ├─ yes → extract_output → END
                  │                         └─ no  → agent (loop)
                  └─ no  → extract_output → END
```

## Setup

```bash
# From repo root
cd misc/agentic

# Create .env from template
cp .env.example .env
# Edit .env with your HF_TOKEN, OLLAMA_URL, etc.

# Install dependencies
uv sync
```

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS` | `["xingyaow/codeact-agent-mistral"]` | JSON list of Ollama model names |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `HF_TOKEN` | — | HuggingFace token for dataset download |
| `TEST_CASE_TIMEOUT` | `120` | Per-task timeout in seconds |
| `MAX_TEST_CASES` | — | Limit number of test cases (omit for all) |
| `OFFSET` | `0` | Skip first N test cases |
| `MAX_EXECUTION_TURNS` | `3` | Max code execution turns per task |

## Running

```bash
uv run agentic-eval
```

Results are saved to `./results/` as CSV files:
- `<model>_async_agent_results_validation.csv` — JITGen (async) strategy
- `<model>_sync_agent_results_validation.csv` — Sync strategy

## Output Columns

| Column | Description |
|--------|-------------|
| `RowID` | MBPP task ID |
| `ErrorOccurred` | Whether execution failed |
| `ExecutionOutput` | Raw stdout captured |
| `HasTimedOut` | Whether execution timed out |
| `ExecutionError` | Error message if any |
| `ExecutionTime` | Total wall-clock time (seconds) |
| `ExecutionTurns` | Number of agent execution turns used |
| `ExpectedOutput` | Ground-truth expected output |
| `ActualOutput` | Cleaned stdout output |
| `CorrectOutput` | Whether actual matches expected |
