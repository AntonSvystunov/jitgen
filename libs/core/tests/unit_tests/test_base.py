"""Tests for jitgen_core.base module."""

from jitgen_core.base import ExecutionResult


def test_execution_result_success():
    result = ExecutionResult(success=True, output="Hello, World!\n")
    assert result.success is True
    assert result.output == "Hello, World!\n"
    assert result.error is None
    assert result.has_timed_out is False


def test_execution_result_failure():
    result = ExecutionResult(
        success=False, error="NameError: name 'x' is not defined"
    )
    assert result.success is False
    assert result.error == "NameError: name 'x' is not defined"
    assert result.has_timed_out is False


def test_execution_result_timeout():
    result = ExecutionResult(success=False, has_timed_out=True)
    assert result.success is False
    assert result.has_timed_out is True


def test_execution_result_defaults():
    result = ExecutionResult(success=False)
    assert result.output is None
    assert result.error is None
    assert result.has_timed_out is False
