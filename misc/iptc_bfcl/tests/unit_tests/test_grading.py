import json
from typing import Any

import pytest
from conftest import load_entry

from iptc_bfcl.grading import ast_checker, grade_calls, standardize_string
from iptc_bfcl.metrics import ToolCallRecord


def _record(
    name: str, turn: int = 0, ok: bool = True, **arguments: Any
) -> ToolCallRecord:
    return ToolCallRecord(
        name,
        0.0,
        0.1,
        ok=ok,
        arguments=json.dumps(arguments, sort_keys=True),
        turn=turn,
    )


def _check(entry_id: str, *calls: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    entry = load_entry(entry_id)
    verdict = ast_checker(
        list(entry.functions), list(calls), list(entry.ground_truth), entry.category
    )
    return verdict.valid, verdict.error_type


def test_strings_are_compared_loosely():
    assert standardize_string("April 1, 2024") == standardize_string("april 1 2024")


@pytest.mark.parametrize(
    ("arguments", "valid", "error_type"),
    [
        ({"base": 10, "height": 5}, True, ""),  # `unit` is optional ("")
        ({"base": 10, "height": 5, "unit": "Units"}, True, ""),
        ({"base": 10, "height": 6}, False, "value_error:others"),
        ({"base": 10}, False, "simple_function_checker:missing_required"),
        ({"base": "10", "height": 5}, False, "type_error:simple"),
        ({"base": 10, "height": 5, "unit": "cm"}, False, "value_error:string"),
    ],
)
def test_simple(arguments, valid, error_type):
    assert _check("simple_0", {"calculate_triangle_area": arguments}) == (
        valid,
        error_type,
    )


def test_simple_needs_exactly_one_call():
    call = {"calculate_triangle_area": {"base": 10, "height": 5}}

    assert _check("simple_0", call, call) == (
        False,
        "simple_function_checker:wrong_count",
    )


def test_dict_parameters_are_checked_key_by_key():
    base = {"database_name": "StudentDB", "table_name": "students"}
    good = {"department": "science", "school": "Bluebird HS"}

    assert _check("simple_89", {"db_fetch_records": {**base, "conditions": good}})[0]
    assert _check(
        "simple_89",
        {"db_fetch_records": {**base, "conditions": {**good, "school": "Other"}}},
    ) == (False, "value_error:dict_value")


def test_multiple_picks_the_expected_function():
    entry = load_entry("multiple_0")
    [answer] = entry.ground_truth
    [(name, params)] = answer.items()
    arguments = {key: values[0] for key, values in params.items() if values[0] != ""}

    assert _check("multiple_0", {name: arguments}) == (True, "")
    other = next(f["name"] for f in entry.functions if f["name"] != name)
    assert not _check("multiple_0", {other: arguments})[0]


def test_parallel_calls_match_in_any_order():
    swift = {"spotify.play": {"artist": "Taylor Swift", "duration": 20}}
    maroon = {"spotify.play": {"artist": "Maroon 5", "duration": 15}}

    assert _check("parallel_0", maroon, swift) == (True, "")
    assert _check("parallel_0", swift)[1].endswith("wrong_count")
    assert not _check("parallel_0", swift, swift)[0]


def test_an_int_is_accepted_for_a_float():
    # JavaScript has no int/float distinction: `7.0` reaches Python as `7`.
    calls = [
        {"area_circle.calculate": {"radius": 5}},
        {"area_rectangle.calculate": {"length": 7, "breadth": 3}},
    ]

    assert _check("parallel_multiple_1", *calls) == (True, "")


def test_grade_counts_successful_calls_once_across_turns():
    entry = load_entry("parallel_0")
    records = [
        _record("spotify.play", turn=0, artist="Taylor Swift", duration=20),
        _record("spotify.play", turn=0, ok=False, artist="Maroon 5"),
        # The retry repeats the call that had already succeeded.
        _record("spotify.play", turn=1, artist="Taylor Swift", duration=20),
        _record("spotify.play", turn=1, artist="Maroon 5", duration=15),
    ]

    grade = grade_calls(entry, records, first_turn=0)

    assert grade.correct
    assert not grade.first_turn_correct
    assert grade.successful_calls == 2
    assert grade.reason == ""


def test_grade_explains_a_wrong_run():
    grade = grade_calls(load_entry("simple_0"), [], first_turn=None)

    assert not grade.correct
    assert not grade.first_turn_correct
    assert grade.reason.startswith("simple_function_checker:wrong_count")


def test_a_failed_first_response_is_not_first_turn_correct():
    # Turn 0 called a tool, but its code failed before any call succeeded.
    records = [_record("calculate_triangle_area", turn=1, base=10, height=5)]

    grade = grade_calls(load_entry("simple_0"), records, first_turn=0)

    assert grade.correct
    assert not grade.first_turn_correct
