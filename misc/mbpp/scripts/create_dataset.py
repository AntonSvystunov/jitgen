import ast
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CACHE_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = SCRIPT_DIR / "dataset"
OUTPUT_FILE_EXAMPLES = OUTPUT_DIR / "mbpp_jitgen_validation.csv"

MIN_ASSERTIONS_REQUIRED = 2

_SAFE_EVAL_GLOBALS: dict[str, Any] = {
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "round": round,
    "sys": SimpleNamespace(getsizeof=sys.getsizeof),
    "__builtins__": {},
}


def parse_assert_statement(assert_stmt: str) -> tuple[str, str] | tuple[None, None]:
    """Split an `assert x == y` statement into its call and expected result.

    Args:
        assert_stmt: A single assertion statement, e.g. `"assert f(1) == 2"`.

    Returns:
        A `(function_call, expected_result)` tuple, or `(None, None)` if the
        statement isn't a simple equality assertion.
    """
    stmt = assert_stmt.strip().replace("assert ", "")
    if " == " not in stmt:
        return None, None
    function_call, expected_result = stmt.split(" == ", 1)
    return function_call.strip(), expected_result.strip()


def extract_function_parameters(function_call: str) -> str:
    """Extract the argument list from a function call expression.

    Args:
        function_call: A call expression, e.g. `"f(1, 2)"`.

    Returns:
        The raw text between the outermost parentheses, or `function_call`
        unchanged if it isn't a call expression.
    """
    match = re.match(r"(\w+)\((.*)\)", function_call)
    if not match:
        return function_call
    return match.group(2)


def safe_eval(expression: str) -> str | None:
    """Evaluate the expected-result expression from an MBPP test assertion.

    Falls back from `ast.literal_eval` to a restricted `eval` for expressions
    that use simple builtins (e.g. `len(...)`), since MBPP's expected results
    aren't always pure literals.

    Args:
        expression: The right-hand side of an `assert ... == expression`.

    Returns:
        The `repr()` of the evaluated expression, or `None` if it can't be
        evaluated. `repr()` (rather than `str()`) is used so string results
        round-trip through the output CSV unambiguously: a bare `str()`
        result like `None` or `NA` is indistinguishable on disk from a
        missing cell, since those are also pandas' default `read_csv` NA
        sentinels — wrapping it in real quote characters (`'None'`) avoids
        the collision. Consumers should recover the value with
        `ast.literal_eval()`. Returning the un-evaluated expression text on
        failure would silently write wrong data (e.g. the literal string
        `"sys.getsizeof(...)"` as the expected output) into the dataset, so
        failures must be propagated to the caller and dropped instead.
    """
    try:
        result = ast.literal_eval(expression)
    except (ValueError, SyntaxError):
        try:
            result = eval(expression, _SAFE_EVAL_GLOBALS)
        except Exception:  # noqa: BLE001 - eval() on arbitrary dataset text can raise anything
            return None
    return repr(result)


def modify_task_text(text: str) -> str:
    """Strip the leading "Write a function to"-style prefix from a task description.

    Args:
        text: The original MBPP task description.

    Returns:
        The text after the first `"to"`, capitalized, or `text` unchanged if
        it contains no `"to"`.
    """
    parts = text.split("to", 1)
    if len(parts) == 2:
        return parts[1].strip().capitalize()
    return text


def create_dataset_with_examples(original_dataset: pd.DataFrame) -> pd.DataFrame:
    """Expand each MBPP task's assertions into example/test row pairs.

    The first parsed assertion becomes the worked example shown alongside
    each remaining assertion, which becomes that row's test case.

    Args:
        original_dataset: The MBPP validation split as a `DataFrame`, with
            `task_id`, `text`, and `test_list` columns.

    Returns:
        A `DataFrame` with one row per (example, test) pair, for tasks with
        at least `MIN_ASSERTIONS_REQUIRED` parseable assertions.
    """
    new_rows: list[dict[str, Any]] = []
    for _, row in original_dataset.iterrows():
        parsed_assertions = []
        for assert_stmt in row["test_list"]:
            function_call, expected_result = parse_assert_statement(assert_stmt)
            if not function_call or not expected_result:
                continue
            test_output = safe_eval(expected_result)
            if test_output is None:
                continue
            parsed_assertions.append(
                {
                    "test_input": extract_function_parameters(function_call),
                    "test_output": test_output,
                }
            )
        if len(parsed_assertions) < MIN_ASSERTIONS_REQUIRED:
            continue

        text = modify_task_text(row["text"])
        example = parsed_assertions[0]
        for assertion in parsed_assertions[1:]:
            new_rows.append(
                {
                    "task_id": row["task_id"],
                    "text": text,
                    "example_test_input": example["test_input"],
                    "example_test_output": example["test_output"],
                    "test_input": assertion["test_input"],
                    "test_output": assertion["test_output"],
                }
            )
    return pd.DataFrame(new_rows)


def main() -> None:
    dataset_full = load_dataset(
        "google-research-datasets/mbpp",
        cache_dir=str(DATA_CACHE_DIR),
        download_mode="reuse_dataset_if_exists",
    )
    df_original = dataset_full["validation"].to_pandas()

    examples_dataset = create_dataset_with_examples(df_original)  # type: ignore

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    examples_dataset.to_csv(OUTPUT_FILE_EXAMPLES, index=False)


if __name__ == "__main__":
    main()
