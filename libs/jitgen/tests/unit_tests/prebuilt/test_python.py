"""Tests for jitgen.prebuilt.python module."""

import pytest

from jitgen.prebuilt.python import (
    create_python_async_jitgen_session,
    create_python_jitgen,
    create_python_jitgen_session,
)
from jitgen_core.v2.aio import AsyncJITGenSession
from jitgen_core.jit import JITGen
from jitgen_core.v2 import JITGenSession
from jitgen.executors.python import InProcPythonExecutor


def test_create_python_jitgen():
    """Test the create_python_jitgen factory function."""
    jitgen = create_python_jitgen()
    
    assert isinstance(jitgen, JITGen)
    assert jitgen.interpreter_type == InProcPythonExecutor
    assert len(jitgen.indentation_tokens) == 3
    assert "_DEDENT" in jitgen.indentation_tokens
    assert "_NEWLINE" in jitgen.indentation_tokens
    assert "$END" in jitgen.indentation_tokens
    assert jitgen.parser is not None


def test_create_python_jitgen_session():
    session = create_python_jitgen_session()

    assert isinstance(session, JITGenSession)
    assert session.interpreter_type == InProcPythonExecutor
    assert session.start_marker == "```python"
    assert session.end_marker == "```"


def test_create_python_async_jitgen_session():
    session = create_python_async_jitgen_session()

    assert isinstance(session, AsyncJITGenSession)
    assert session.interpreter_type == InProcPythonExecutor
    assert session.start_marker == "```python"
    assert session.end_marker == "```"


def test_create_python_jitgen_session_with_tools():
    def add(left: int, right: int) -> int:
        return left + right

    session = create_python_jitgen_session(tools={"add": add})

    output = session.push("```python\nprint(add(2, 3))\n```")

    assert output == "5\n"


@pytest.mark.asyncio
async def test_create_python_async_jitgen_session_with_tools():
    def add(left: int, right: int) -> int:
        return left + right

    stdout_events: list[str] = []
    session = create_python_async_jitgen_session(tools={"add": add})

    @session.on_stdout
    async def capture_stdout(output: str) -> None:
        stdout_events.append(output)

    await session.apush("```python\nprint(add(2, 3))\n```")
    await session.aflush()

    assert stdout_events == ["5\n"]

