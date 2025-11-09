"""Tests for jitgen.core.jit module."""

import pytest
from collections.abc import Iterator
from lark import Lark
from lark.indenter import PythonIndenter

from jitgen.core.jit import JITGen
from jitgen.executors.python import InProcPythonExecutor


@pytest.fixture
def python_jitgen():
    """Create a JITGen instance configured for Python."""
    python_parser = Lark.open_from_package(
        "lark", "python.lark", ["grammars"],
        parser="lalr",
        postlex=PythonIndenter(),
        start="file_input",
        propagate_positions=True,
    )
    return JITGen(
        parser=python_parser,
        interpreter_type=InProcPythonExecutor,
        indentation_tokens={"_DEDENT", "_NEWLINE", "$END"},
    )


def test_run_from_stream_single_statement(python_jitgen: JITGen):
    """Test running a single complete statement."""
    def code_stream() -> Iterator[str]:
        yield "print('Hello')\n"
    
    results = list(python_jitgen.run_from_stream(code_stream()))
    assert len(results) == 1
    assert results[0] == "Hello\n"


def test_run_from_stream_multiple_statements(python_jitgen: JITGen):
    """Test running multiple complete statements."""
    def code_stream() -> Iterator[str]:
        yield "print('Hello')\n"
        yield "print('World')\n"
    
    results = list(python_jitgen.run_from_stream(code_stream()))
    assert len(results) == 2
    assert results[0] == "Hello\n"
    assert results[1] == "World\n"


def test_run_from_stream_incremental(python_jitgen: JITGen):
    """Test running code that arrives incrementally."""
    def code_stream() -> Iterator[str]:
        yield "a = "
        yield "10\n"
        yield "print(a)\n"
    
    results = list(python_jitgen.run_from_stream(code_stream()))
    # First result is empty (from "a = 10\n" assignment), second is "10\n" (from print)
    assert len(results) == 2
    assert results[0] == ""  # Assignment produces no output
    assert results[1] == "10\n"


def test_run_from_stream_multiple_statements_in_one_chunk(python_jitgen: JITGen):
    """Test running multiple statements that arrive in one chunk."""
    def code_stream() -> Iterator[str]:
        yield "print('First')\nprint('Second')\n"
    
    results = list(python_jitgen.run_from_stream(code_stream()))
    assert len(results) == 2
    assert results[0] == "First\n"
    assert results[1] == "Second\n"


def test_run_from_stream_incomplete_statement(python_jitgen: JITGen):
    """Test handling incomplete statements (should wait for more)."""
    def code_stream() -> Iterator[str]:
        yield "print("
        yield "'Hello')\n"
    
    results = list(python_jitgen.run_from_stream(code_stream()))
    assert len(results) == 1
    assert results[0] == "Hello\n"


def test_run_from_stream_syntax_error(python_jitgen: JITGen):
    """Test that syntax errors halt processing."""
    def code_stream() -> Iterator[str]:
        yield "print('Hello')\n"
        yield "print('World'  # Missing closing paren\n"
    
    with pytest.raises(ValueError, match="Syntax error"):
        _ = list(python_jitgen.run_from_stream(code_stream()))


def test_run_from_stream_runtime_error(python_jitgen: JITGen):
    """Test that runtime errors halt processing."""
    def code_stream() -> Iterator[str]:
        yield "print('Hello')\n"
        yield "print(undefined_variable)\n"
    
    with pytest.raises(ValueError, match="Error detected"):
        _ = list(python_jitgen.run_from_stream(code_stream()))


def test_run_from_stream_empty_buffer(python_jitgen: JITGen):
    """Test handling empty buffer."""
    def code_stream() -> Iterator[str]:
        yield ""
    
    results = list(python_jitgen.run_from_stream(code_stream()))
    assert len(results) == 0


def test_run_from_stream_whitespace_only(python_jitgen: JITGen):
    """Test handling whitespace-only buffer."""
    def code_stream() -> Iterator[str]:
        yield "   \n   \n"
    
    results = list(python_jitgen.run_from_stream(code_stream()))
    assert len(results) == 0


def test_run_from_stream_indentation_incomplete(python_jitgen: JITGen):
    """Test handling incomplete indented blocks."""
    def code_stream() -> Iterator[str]:
        yield "if True:\n"
        yield "    print('Hello')\n"
    
    results = list(python_jitgen.run_from_stream(code_stream()))
    assert len(results) == 1
    assert results[0] == "Hello\n"


@pytest.mark.asyncio
async def test_arun_from_stream_single_statement(python_jitgen: JITGen):
    """Test async running a single complete statement."""
    async def code_stream():
        yield "print('Hello')\n"
    
    results = []
    async for result in python_jitgen.arun_from_stream(code_stream()):
        results.append(result)
    
    assert len(results) == 1
    assert results[0] == "Hello\n"


@pytest.mark.asyncio
async def test_arun_from_stream_multiple_statements(python_jitgen: JITGen):
    """Test async running multiple complete statements."""
    async def code_stream():
        yield "print('Hello')\n"
        yield "print('World')\n"
    
    results = []
    async for result in python_jitgen.arun_from_stream(code_stream()):
        results.append(result)
    
    assert len(results) == 2
    assert results[0] == "Hello\n"
    assert results[1] == "World\n"


@pytest.mark.asyncio
async def test_arun_from_stream_incremental(python_jitgen: JITGen):
    """Test async running code that arrives incrementally."""
    async def code_stream():
        yield "a = "
        yield "10\n"
        yield "print(a)\n"
    
    results = []
    async for result in python_jitgen.arun_from_stream(code_stream()):
        results.append(result)
    
    # First result is empty (from "a = 10\n" assignment), second is "10\n" (from print)
    assert len(results) == 2
    assert results[0] == ""  # Assignment produces no output
    assert results[1] == "10\n"


@pytest.mark.asyncio
async def test_arun_from_stream_syntax_error(python_jitgen: JITGen):
    """Test that async syntax errors halt processing."""
    async def code_stream():
        yield "print('Hello')\n"
        yield "print('World'  # Missing closing paren\n"
    
    with pytest.raises(ValueError, match="Syntax error"):
        async for _ in python_jitgen.arun_from_stream(code_stream()):
            pass


@pytest.mark.asyncio
async def test_arun_from_stream_runtime_error(python_jitgen: JITGen):
    """Test that async runtime errors halt processing."""
    async def code_stream():
        yield "print('Hello')\n"
        yield "print(undefined_variable)\n"
    
    with pytest.raises(ValueError, match="Error detected"):
        async for _ in python_jitgen.arun_from_stream(code_stream()):
            pass


@pytest.mark.asyncio
async def test_arun_from_stream_empty_buffer(python_jitgen: JITGen):
    """Test async handling empty buffer."""
    async def code_stream():
        yield ""
    
    results = []
    async for result in python_jitgen.arun_from_stream(code_stream()):
        results.append(result)
    
    assert len(results) == 0


def test_execute_statement_with_empty_statement(python_jitgen: JITGen):
    """Test _execute_statement with statement that has no meta."""
    # Test with a statement that has meta=None (edge case)
    # We'll use a real parse tree but test the meta=None path by mocking
    from unittest.mock import Mock
    
    mock_stmt = Mock()
    mock_stmt.meta = None
    
    result = python_jitgen._execute_statement("", mock_stmt, InProcPythonExecutor())
    assert result == ""


@pytest.mark.asyncio
async def test_aexecute_statement_with_empty_statement(python_jitgen: JITGen):
    """Test _aexecute_statement with statement that has no meta."""
    # Test with a statement that has meta=None (edge case)
    from unittest.mock import Mock
    
    mock_stmt = Mock()
    mock_stmt.meta = None
    
    result = await python_jitgen._aexecute_statement("", mock_stmt, InProcPythonExecutor())
    assert result == ""


def test_run_from_stream_flush_remaining_code(python_jitgen: JITGen):
    """Test that remaining code is flushed after stream ends."""
    def code_stream() -> Iterator[str]:
        yield "print('First')\n"
        yield "print('Second')\n"  # Complete statement with newline
    
    results = list(python_jitgen.run_from_stream(code_stream()))
    assert len(results) == 2
    assert results[0] == "First\n"
    assert results[1] == "Second\n"


@pytest.mark.asyncio
async def test_arun_from_stream_flush_remaining_code(python_jitgen: JITGen):
    """Test that remaining code is flushed after async stream ends."""
    async def code_stream():
        yield "print('First')\n"
        yield "print('Second')\n"  # Complete statement with newline
    
    results = []
    async for result in python_jitgen.arun_from_stream(code_stream()):
        results.append(result)
    
    assert len(results) == 2
    assert results[0] == "First\n"
    assert results[1] == "Second\n"


def test_run_from_stream_flush_with_syntax_error(python_jitgen: JITGen):
    """Test that syntax errors in flush raise ValueError."""
    def code_stream() -> Iterator[str]:
        yield "print('Hello')"
        yield "  invalid syntax"  # Incomplete/invalid
    
    with pytest.raises(ValueError, match="Syntax error"):
        _ = list(python_jitgen.run_from_stream(code_stream()))


@pytest.mark.asyncio
async def test_arun_from_stream_flush_with_syntax_error(python_jitgen: JITGen):
    """Test that syntax errors in async flush raise ValueError."""
    async def code_stream():
        yield "print('Hello')"
        yield "  invalid syntax"  # Incomplete/invalid
    
    with pytest.raises(ValueError, match="Syntax error"):
        async for _ in python_jitgen.arun_from_stream(code_stream()):
            pass

