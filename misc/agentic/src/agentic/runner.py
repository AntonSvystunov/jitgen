import asyncio
from pathlib import Path

from tqdm.asyncio import tqdm

from agentic.agents import AgentProtocol, ExecutionResult, IncrementalAgent, StandardAgent
from agentic.config import Config
from agentic.data import load_context_files, load_tasks_dataset
from agentic.executors import OpenSandboxPythonExecutor
from agentic.models import get_model
from agentic.results import make_run_stem, result_to_row, write_csv, write_jsonl_row
from jitgen.executors.python import InProcPythonExecutor


def _make_agent(
    cfg: Config,
    model,
    executor,
    context_file_names: list[str],
) -> AgentProtocol:
    if cfg.mode == "incremental":
        return IncrementalAgent(
            model=model,
            executor=executor,
            context_file_names=context_file_names,
            max_steps=cfg.max_steps,
        )
    return StandardAgent(
        model=model,
        executor=executor,
        context_file_names=context_file_names,
        max_steps=cfg.max_steps,
    )


async def run_benchmark(cfg: Config) -> None:
    dataset = load_tasks_dataset(cfg.dataset)
    context_file_names = load_context_files(cfg.context_data_directory)

    results_dir = Path(cfg.results_dir)
    stem = make_run_stem(cfg.model_name, cfg.mode, cfg.dataset)
    jsonl_path = results_dir / "raw" / f"{stem}.jsonl"
    csv_path = results_dir / "tables" / f"{stem}.csv"

    rows: list[dict] = []

    print(f"Model : {cfg.model_name}")
    print(f"Mode  : {cfg.mode}")
    print(f"Tasks : {len(dataset)}")
    print(f"JSONL : {jsonl_path}")
    print(f"CSV   : {csv_path}")
    print()

    async with get_model(
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        seed=cfg.seed,
    ) as model:
        for task in tqdm(dataset, desc="tasks"):
            task_id: str = task["task_id"]
            question: str = task["question"]
            guidelines: str = task["guidelines"]
            expected_answer: str | None = task.get("answer")

            if cfg.executor == "sandbox":
                async with OpenSandboxPythonExecutor(
                    image=cfg.sandbox_image,
                    domain=cfg.sandbox_domain,
                    packages=cfg.sandbox_packages,
                ) as executor:
                    agent = _make_agent(cfg, model, executor, context_file_names)
                    result: ExecutionResult = await agent.run(question, guidelines)
            else:
                executor = InProcPythonExecutor(tools={"open": open})
                agent = _make_agent(cfg, model, executor, context_file_names)
                result = await agent.run(question, guidelines)

            write_jsonl_row(jsonl_path, result, task_id, expected_answer, cfg.mode)
            row = result_to_row(result, task_id, expected_answer, cfg.mode)
            rows.append(row)

            status = "OK" if row["is_correct"] else ("TIMEOUT" if row["is_timeout"] else "WRONG")
            print(f"  [{status}] {task_id}  steps={row['steps_count']}  time={row['total_time']:.1f}s  tokens={row['total_tokens']}")

    write_csv(csv_path, rows)
    print(f"\nWrote {len(rows)} rows → {csv_path}")
