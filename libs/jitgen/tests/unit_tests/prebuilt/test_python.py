"""Tests for jitgen.prebuilt.python module."""

import pytest

from jitgen.executors.python import InProcPythonExecutor
from jitgen.prebuilt.python import create_python_jitgen
from jitgen_core import Session


def test_create_python_jitgen_returns_session():
    session = create_python_jitgen()
    assert isinstance(session, Session)


def test_create_python_jitgen_with_default_executor():
    """No executor arg → a stock InProcPythonExecutor is used."""
    session = create_python_jitgen()
    assert session._executor is not None  # noqa: SLF001
    assert isinstance(session._executor, InProcPythonExecutor)  # noqa: SLF001


def test_create_python_jitgen_accepts_custom_executor():
    custom = InProcPythonExecutor(timeout=5.0)
    session = create_python_jitgen(executor=custom)
    assert session._executor is custom  # noqa: SLF001


@pytest.mark.asyncio
async def test_create_python_jitgen_executes_simple_statement():
    session = create_python_jitgen()
    session.push("print('hello')\n")
    out = await session.result()
    assert out == "hello\n"


@pytest.mark.asyncio
async def test_create_python_jitgen_maintains_repl_state():
    session = create_python_jitgen()
    session.push("x = 42\n")
    session.push("print(x)\n")
    out = await session.result()
    assert out == "42\n"


@pytest.mark.asyncio
async def test_create_python_jitgen_with_injected_tool():
    exec_ = InProcPythonExecutor(tools={"add": lambda a, b: a + b})
    session = create_python_jitgen(executor=exec_)
    session.push("print(add(2, 3))\n")
    out = await session.result()
    assert out == "5\n"
