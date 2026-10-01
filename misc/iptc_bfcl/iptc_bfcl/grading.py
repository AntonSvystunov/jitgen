# The checker below is adapted from BFCL's AST checker,
# https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/
# bfcl_eval/eval_checker/ast_eval/ast_checker.py
# Copyright the Gorilla authors, licensed under the Apache License, Version 2.0
# (https://www.apache.org/licenses/LICENSE-2.0). Changes: only the Python-language
# path is kept (every arm's calls are graded as Python values, JavaScript's
# included); function names are compared as BFCL spells them, since calls are
# recorded under their BFCL name; the result is a `Verdict` instead of a dict; a
# schema without `type` or `items` is treated as `any`; type annotations added.
import json
import re
from dataclasses import dataclass
from typing import Any

from iptc_bfcl.dataset import BfclEntry
from iptc_bfcl.metrics import ToolCallRecord

# BFCL type -> the Python type its value must have.
PYTHON_TYPE_MAPPING: dict[str, type] = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}
# Types whose items are type checked too.
PYTHON_NESTED_TYPE_CHECK_LIST = ("array", "tuple")

Call = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Verdict:
    """Whether a set of calls matches the accepted answers, and why not."""

    valid: bool
    error: str = ""
    error_type: str = ""


_OK = Verdict(True)


def _fail(error_type: str, error: str) -> Verdict:
    return Verdict(False, error, error_type)


def _find_description(functions: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for function in functions:
        if function["name"] == name:
            return function
    msg = f"no function named {name!r}"
    raise KeyError(msg)


def _possible_answer_type(possible_answer: list[Any]) -> type | None:
    for answer in possible_answer:
        if answer != "":  # "" marks an optional parameter
            return type(answer)
    return None


def _type_checker(
    param: str,
    value: Any,
    possible_answer: list[Any],
    expected_type_description: str,
    expected_type: type,
    nested_type: type | None,
) -> tuple[Verdict, bool]:
    """Check `value`'s type, one level deep for lists.

    Returns:
        The verdict, and whether the answer treats the value as a variable
        (a value of another type than the declared one, e.g. a string naming
        a variable).
    """
    is_variable = False
    possible_answer_type = _possible_answer_type(possible_answer)
    if possible_answer_type is not None and possible_answer_type != expected_type:
        is_variable = True

    if type(value) is expected_type:
        if nested_type is None:
            return _OK, is_variable
        for possible_answer_item in possible_answer:
            flag = True
            if type(possible_answer_item) is list:
                for value_item in value:
                    verdict, _ = _type_checker(
                        param,
                        value_item,
                        possible_answer_item,
                        nested_type.__name__,
                        nested_type,
                        None,
                    )
                    if not verdict.valid:
                        flag = False
                        break
            if flag:
                return _OK, is_variable
        return _fail(
            "type_error:nested",
            f"Nested type checking failed for parameter {param!r}. Expected outer "
            f"type {expected_type_description} with inner type "
            f"{nested_type.__name__}. Parameter value: {value!r}.",
        ), is_variable

    if possible_answer_type is not None and type(value) is possible_answer_type:
        return _OK, True
    return _fail(
        "type_error:simple",
        f"Incorrect type for parameter {param!r}. Expected type "
        f"{expected_type_description}, got {type(value).__name__}. "
        f"Parameter value: {value!r}.",
    ), is_variable


def standardize_string(input_string: str) -> str:
    """Normalize a string for comparison, as BFCL does.

    Spaces and `,./-_*^` are removed, the string is lowercased, and single
    quotes become double quotes, so `April 1, 2024` matches `april 1 2024`.
    """
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def _standardize(value: Any) -> Any:
    return standardize_string(value) if type(value) is str else value


def _string_checker(param: str, value: str, possible_answer: list[Any]) -> Verdict:
    accepted = [standardize_string(a) for a in possible_answer if type(a) is str]
    if standardize_string(value) not in accepted:
        return _fail(
            "value_error:string",
            f"Invalid value for parameter {param!r}: {value!r}. "
            f"Expected one of {possible_answer}. Case insensitive.",
        )
    return _OK


def _list_checker(param: str, value: list[Any], possible_answer: list[Any]) -> Verdict:
    standardized = [_standardize(item) for item in value]
    accepted = [
        [_standardize(item) for item in answer]
        for answer in possible_answer
        if isinstance(answer, list)
    ]
    if standardized not in accepted:
        return _fail(
            "value_error:list/tuple",
            f"Invalid value for parameter {param!r}: {value!r}. "
            f"Expected one of {possible_answer}.",
        )
    return _OK


def _dict_checker(
    param: str, value: dict[str, Any], possible_answers: list[Any]
) -> Verdict:
    # Like BFCL's: handles dictionaries one level deep, which the dataset needs.
    result = _fail("dict_checker:unclear", f"No accepted value for {param!r}.")
    for possible_answer in possible_answers:
        if possible_answer == "":
            continue
        result = _dict_matches(value, possible_answer)
        if result.valid:
            return _OK
    return result


def _dict_matches(value: dict[str, Any], possible_answer: dict[str, Any]) -> Verdict:
    for key, item in value.items():
        if key not in possible_answer:
            return _fail(
                "value_error:dict_key", f"Unexpected dict key parameter: {key!r}."
            )
        accepted = [_standardize(a) for a in possible_answer[key]]
        if _standardize(item) not in accepted:
            return _fail(
                "value_error:dict_value",
                f"Invalid value for parameter {key!r}: {item!r}. "
                f"Expected one of {accepted}.",
            )
    for key, accepted in possible_answer.items():
        if key not in value and "" not in accepted:
            return _fail(
                "value_error:dict_key", f"Missing dict key parameter: {key!r}."
            )
    return _OK


def _list_dict_checker(
    param: str, value: list[Any], possible_answers: list[Any]
) -> Verdict:
    # The dictionaries must come in the accepted answer's order.
    result = _fail("list_dict_checker:unclear", f"No accepted value for {param!r}.")
    for possible_answer in possible_answers:
        if not isinstance(possible_answer, list) or len(value) != len(possible_answer):
            result = _fail(
                "value_error:list_dict_count",
                "Wrong number of dictionaries in the list.",
            )
            continue
        for item, accepted in zip(value, possible_answer):
            result = _dict_checker(param, item, [accepted])
            if not result.valid:
                break
        else:
            return _OK
    return result


def _expected_types(spec: dict[str, Any]) -> tuple[str, type, type | None]:
    description = spec.get("type", "any")
    expected = PYTHON_TYPE_MAPPING.get(description, str)
    nested = None
    if description in PYTHON_NESTED_TYPE_CHECK_LIST:
        nested_description = (spec.get("items") or {}).get("type", "any")
        nested = PYTHON_TYPE_MAPPING.get(nested_description, str)
    return description, expected, nested


def _check_param(
    param: str, value: Any, spec: dict[str, Any], possible_answer: list[Any]
) -> Verdict:
    """Check one parameter's type and value against its accepted values."""
    description, expected, nested = _expected_types(spec)
    if description == "tuple" and type(value) is tuple:
        value = list(value)
    # Python converts int to float; JavaScript has no int/float distinction.
    if description == "float" and type(value) is int:
        value = float(value)

    verdict, is_variable = _type_checker(
        param, value, possible_answer, description, expected, nested
    )
    if not verdict.valid:
        return verdict
    if not is_variable:
        if expected is dict:
            return _dict_checker(param, value, possible_answer)
        if expected is list and nested is dict:
            return _list_dict_checker(param, value, possible_answer)
        if expected is str:
            return _string_checker(param, value, possible_answer)
        if expected is list:
            return _list_checker(param, value, possible_answer)
    if value not in possible_answer:
        return _fail(
            "value_error:others",
            f"Invalid value for parameter {param!r}: {value!r}. "
            f"Expected one of {possible_answer}.",
        )
    return _OK


def simple_function_checker(
    function: dict[str, Any], call: Call, possible_answer: Call
) -> Verdict:
    """Check one call against one accepted answer.

    Args:
        function: The BFCL doc of the expected function.
        call: `{name: arguments}` as made.
        possible_answer: `{name: {param: [accepted values]}}`.

    Returns:
        Whether the call is accepted.
    """
    accepted = next(iter(possible_answer.values()))
    name = function["name"]
    params = function.get("parameters", {}).get("properties", {})
    required = function.get("parameters", {}).get("required", [])
    if name not in call:
        return _fail(
            "simple_function_checker:wrong_func_name",
            f"Function name {name!r} not found in model output.",
        )
    arguments = call[name]
    for param in required:
        if param not in arguments:
            return _fail(
                "simple_function_checker:missing_required",
                f"Missing required parameter: {param!r}.",
            )
    for param, value in arguments.items():
        if param not in params or param not in accepted:
            return _fail(
                "simple_function_checker:unexpected_param",
                f"Unexpected parameter: {param!r}.",
            )
        verdict = _check_param(param, value, params[param], accepted[param])
        if not verdict.valid:
            return verdict
    for param, values in accepted.items():
        if param not in arguments and "" not in values:
            return _fail(
                "simple_function_checker:missing_optional",
                f"Optional parameter {param!r} not provided and not marked as optional.",
            )
    return _OK


def parallel_function_checker_no_order(
    functions: list[dict[str, Any]], calls: list[Call], possible_answers: list[Call]
) -> Verdict:
    """Match every accepted answer to a distinct call, in any order."""
    if len(calls) != len(possible_answers):
        return _fail(
            "parallel_function_checker_no_order:wrong_count",
            f"Wrong number of functions: {len(calls)}, "
            f"expected {len(possible_answers)}.",
        )
    matched: set[int] = set()
    for index, possible_answer in enumerate(possible_answers):
        function = _find_description(functions, next(iter(possible_answer)))
        errors = []
        for call_index, call in enumerate(calls):
            if call_index in matched:
                continue
            verdict = simple_function_checker(function, call, possible_answer)
            if verdict.valid:
                matched.add(call_index)
                break
            errors.append(verdict.error)
        else:
            return _fail(
                "parallel_function_checker_no_order:cannot_find_match",
                f"No call matches expected call {index}: {' | '.join(errors)}",
            )
    return _OK


def multiple_function_checker(
    functions: list[dict[str, Any]], calls: list[Call], possible_answers: list[Call]
) -> Verdict:
    """Check the one call of a `multiple` entry, which picks among functions."""
    if len(calls) != len(possible_answers):
        return _fail(
            "multiple_function_checker:wrong_count",
            f"Wrong number of functions: {len(calls)}, "
            f"expected {len(possible_answers)}.",
        )
    function = _find_description(functions, next(iter(possible_answers[0])))
    return simple_function_checker(function, calls[0], possible_answers[0])


def ast_checker(
    functions: list[dict[str, Any]],
    calls: list[Call],
    possible_answers: list[Call],
    category: str,
) -> Verdict:
    """Check `calls` with the checker BFCL uses for `category`.

    Args:
        functions: The entry's BFCL function docs.
        calls: The calls made, as `{name: arguments}`.
        possible_answers: The entry's ground truth.
        category: The entry's category.

    Returns:
        Whether the calls are accepted.
    """
    if "parallel" in category:
        return parallel_function_checker_no_order(functions, calls, possible_answers)
    if "multiple" in category:
        return multiple_function_checker(functions, calls, possible_answers)
    if len(calls) != 1:
        return _fail(
            "simple_function_checker:wrong_count",
            f"Wrong number of functions: {len(calls)}, expected 1.",
        )
    # Like BFCL: a simple entry's one function, whatever name its answer uses
    # (`simple_363`'s answer says `find_closest` for `restaurant_search.find_closest`).
    return simple_function_checker(functions[0], calls[0], possible_answers[0])


@dataclass(frozen=True)
class Grade:
    """How one run's calls score against the entry's accepted answers.

    Attributes:
        correct: Every successful call of the run, with exact repeats counted
            once, is accepted. The primary metric: calls spread over several
            turns still count, and a retry repeating a call that already
            succeeded doesn't spoil the match.
        first_turn_correct: Only the successful calls of the first turn that
            called a tool are accepted, the closest match to BFCL's own
            scoring of a single response. A first response that failed
            outright (invalid code or JSON, every call rejected) scores false
            even when a retry got it right.
        reason: Why `correct` is false, empty otherwise.
        successful_calls: How many distinct successful calls `correct` saw.
    """

    correct: bool
    first_turn_correct: bool
    reason: str
    successful_calls: int


def _distinct_calls(records: list[ToolCallRecord]) -> list[Call]:
    seen: set[tuple[str, str]] = set()
    calls = []
    for record in records:
        key = (record.name, record.arguments)
        if record.ok and key not in seen:
            seen.add(key)
            calls.append({record.name: json.loads(record.arguments)})
    return calls


def grade_calls(
    entry: BfclEntry, records: list[ToolCallRecord], first_turn: int | None
) -> Grade:
    """Grade a run's recorded calls against the entry's accepted answers.

    Args:
        entry: The entry the run answered.
        records: Every call the run made, failed ones included.
        first_turn: The first turn that called a tool, `None` if none did.

    Returns:
        The run's grade.
    """
    functions, answers = list(entry.functions), list(entry.ground_truth)
    calls = _distinct_calls(records)
    verdict = ast_checker(functions, calls, answers, entry.category)
    first_calls = _distinct_calls([r for r in records if r.turn == first_turn])
    first_verdict = ast_checker(functions, first_calls, answers, entry.category)
    return Grade(
        correct=verdict.valid,
        first_turn_correct=first_verdict.valid,
        reason="" if verdict.valid else f"{verdict.error_type}: {verdict.error}",
        successful_calls=len(calls),
    )
