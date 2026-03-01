"""Tests for jitgen_core.base module."""

from jitgen_core.base import ExecutionResult, SourceCode


def test_execution_result_success():
    """Test ExecutionResult with successful execution."""
    result = ExecutionResult(
        success=True,
        output="Hello, World!\n",
        error=None,
        has_timed_out=False,
        timeout_value=None,
    )
    
    assert result.success is True
    assert result.output == "Hello, World!\n"
    assert result.error is None
    assert result.has_timed_out is False
    assert result.timeout_value is None


def test_execution_result_failure():
    """Test ExecutionResult with failed execution."""
    result = ExecutionResult(
        success=False,
        output="",
        error="NameError: name 'x' is not defined",
        has_timed_out=False,
        timeout_value=None,
    )
    
    assert result.success is False
    assert result.output == ""
    assert result.error == "NameError: name 'x' is not defined"
    assert result.has_timed_out is False
    assert result.timeout_value is None


def test_execution_result_timeout():
    """Test ExecutionResult with timeout."""
    result = ExecutionResult(
        success=False,
        output="",
        error=None,
        has_timed_out=True,
        timeout_value=5.0,
    )
    
    assert result.success is False
    assert result.has_timed_out is True
    assert result.timeout_value == 5.0


def test_execution_result_defaults():
    """Test ExecutionResult with default values."""
    result = ExecutionResult(success=False)
    
    assert result.success is False
    assert result.output is None
    assert result.error is None
    assert result.has_timed_out is False
    assert result.timeout_value is None


def test_source_code_type():
    """Test that SourceCode is a type alias for str."""
    code: SourceCode = "print('test')"
    assert isinstance(code, str)
    assert code == "print('test')"

