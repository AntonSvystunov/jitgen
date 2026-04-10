from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from random import randint
from time import perf_counter

from tqdm import tqdm

from .agents import IncrementalAgentSession, SequentialAgentSession
from .config import AgenticEvaluationConfig, create_model, reinit_model
from .data import AgenticTask, load_context_files, load_tasks
from .results import AgenticEvaluationRecord, write_results

logger = logging.getLogger(__name__)


def _build_session(config: AgenticEvaluationConfig, model, context_files: list[str], task: AgenticTask):
    session_class = (
        SequentialAgentSession
        if config.mode == "sequential"
        else IncrementalAgentSession
    )
    return session_class(
        model=model,
        context_files=context_files,
        question=task.question,
        guidelines=task.guidelines,
        max_steps=config.max_steps,
    )


def _normalize_solution(output: str | None) -> str | None:
    if output is None:
        return None
    stripped = output.strip()
    return stripped or None


async def _run_task(
    config: AgenticEvaluationConfig,
    model,
    context_files: list[str],
    task: AgenticTask,
) -> AgenticEvaluationRecord:
    session = _build_session(config, model, context_files, task)
    started_at = perf_counter()

    try:
        result = await asyncio.wait_for(session.run(), timeout=config.max_timeout)
        final_result = _normalize_solution(result.output)
        expected_answer = task.answer.strip()

        return AgenticEvaluationRecord(
            model_name=config.model_name,
            mode=config.mode,
            task_id=task.task_id,
            success=result.success,
            final_result=final_result,
            expected_answer=task.answer,
            is_correct=final_result == expected_answer if final_result is not None else False,
            is_timeout=False,
            is_error=False,
            no_steps_left=result.steps_left == 0,
            steps_count=result.steps_executed,
            total_time=result.total_time,
            steps_executed=result.steps_executed,
            steps_duration=result.steps_duration,
            messages=result.messages,
        )
    except asyncio.TimeoutError:
        total_time = perf_counter() - started_at
        logger.warning("Task %s timed out after %.2fs", task.task_id, total_time)

        return AgenticEvaluationRecord(
            model_name=config.model_name,
            mode=config.mode,
            task_id=task.task_id,
            success=False,
            final_result=None,
            expected_answer=task.answer,
            is_correct=False,
            is_timeout=True,
            is_error=False,
            no_steps_left=session.steps_left == 0,
            steps_count=session.steps_executed,
            total_time=total_time,
            steps_executed=session.steps_executed,
            steps_duration=session.steps_duration,
            messages=session.get_messages(include_inflight=True),
        )
    except Exception as error:
        total_time = perf_counter() - started_at
        logger.exception("Task %s failed", task.task_id)

        return AgenticEvaluationRecord(
            model_name=config.model_name,
            mode=config.mode,
            task_id=task.task_id,
            success=False,
            final_result=None,
            expected_answer=task.answer,
            is_correct=False,
            is_timeout=False,
            is_error=True,
            no_steps_left=session.steps_left == 0,
            steps_count=session.steps_executed,
            total_time=total_time,
            steps_executed=session.steps_executed,
            steps_duration=session.steps_duration,
            messages=session.get_messages(include_inflight=True),
        )


async def run_evaluation(
    config: AgenticEvaluationConfig,
) -> tuple[str, str]:
    run_started_at = datetime.now()
    session_id = randint(100000, 999999)

    tqdm.write(f"\n{'=' * 60}", file=sys.stderr)
    tqdm.write("Starting agentic evaluation", file=sys.stderr)
    tqdm.write(f"Session ID: {config.session_id}", file=sys.stderr)
    tqdm.write(
        f"Model: {config.model_name} | Mode: {config.mode} | Dataset: {config.dataset}",
        file=sys.stderr,
    )
    tqdm.write(f"{'=' * 60}\n", file=sys.stderr)

    tqdm.write("Loading context files...", file=sys.stderr)
    context_files = load_context_files(config)
    tqdm.write(f"Resolved {len(context_files)} context files", file=sys.stderr)

    tqdm.write("Loading tasks...", file=sys.stderr)
    tasks = load_tasks(config)
    tqdm.write(f"Loaded {len(tasks)} tasks", file=sys.stderr)
    
    await reinit_model(config)

    model = create_model(config)

    tqdm.write("Warming up model...", file=sys.stderr)
    await model.ainvoke("Hi")
    tqdm.write("Model ready", file=sys.stderr)

    records: list[AgenticEvaluationRecord] = []
    for task in tqdm(tasks, desc="Evaluating", unit="task"):
        records.append(await _run_task(config, model, context_files, task))

    csv_path, json_path = write_results(
        records,
        results_directory=config.results_directory,
        model_name=config.model_name,
        mode=config.mode,
        dataset=config.dataset,
        started_at=run_started_at,
    )

    success_count = sum(record.success for record in records)
    timeout_count = sum(record.is_timeout for record in records)
    error_count = sum(record.is_error for record in records)

    tqdm.write(f"Saved table results to {csv_path}", file=sys.stderr)
    tqdm.write(f"Saved raw results to {json_path}", file=sys.stderr)
    tqdm.write(
        f"Completed {len(records)} tasks | successes={success_count} timeouts={timeout_count} errors={error_count}",
        file=sys.stderr,
    )

    return str(csv_path), str(json_path)