from mbpp.dataset import IndexedCase
from mbpp.run_types import RunResult


def print_result(case: IndexedCase, result: RunResult) -> None:
    marker = {"ok": "ok", "timeout": "TIMEOUT", "error": "ERROR"}[result.status]
    first_statement = (
        f"{result.first_statement_seconds:.3f}s"
        if result.first_statement_seconds is not None
        else "n/a"
    )
    print(
        f"[{result.config.model}] [{result.strategy}] "
        f"row={result.dataset_row} task_id={case.example.task_id} "
        f"{marker} elapsed={result.elapsed_seconds:.3f}s first_stmt={first_statement}"
    )
    if result.error_detail:
        print(f"    {result.error_detail}")
