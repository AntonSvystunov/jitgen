"""Unit tests for stateful sync/async JITGen sessions."""

from unittest.mock import Mock

import pytest
from lark import Lark
from lark.tree import Branch

from jitgen_core import AsyncJITGenSession, ExecutionResult, JITGenSession
from jitgen_core.base import BaseExecutor


class MockExecutor(BaseExecutor):
    def execute(self, source_code: str, *, timeout: float = 5.0) -> ExecutionResult:
        return ExecutionResult(success=True, output=f"output:{source_code.strip()}")

    async def aexecute(
        self, source_code: str, *, timeout: float = 5.0
    ) -> ExecutionResult:
        return ExecutionResult(success=True, output=f"output:{source_code.strip()}")


class MockFailingExecutor(BaseExecutor):
    def execute(self, source_code: str, *, timeout: float = 5.0) -> ExecutionResult:
        return ExecutionResult(success=False, error="Execution failed", output="")

    async def aexecute(
        self, source_code: str, *, timeout: float = 5.0
    ) -> ExecutionResult:
        return ExecutionResult(success=False, error="Execution failed", output="")


def create_mock_statement(start_pos: int, end_pos: int) -> Branch:
    statement = Mock(spec=Branch)
    meta = Mock()
    meta.start_pos = start_pos
    meta.end_pos = end_pos
    statement.meta = meta
    return statement


def create_mock_tree(*statements: Branch) -> Branch:
    tree = Mock(spec=Branch)
    tree.children = list(statements)
    return tree


@pytest.fixture
def mock_parser() -> Mock:
    parser = Mock(spec=Lark)

    def parse_side_effect(buffer: str) -> Branch:
        return create_mock_tree(create_mock_statement(0, len(buffer)))

    parser.parse.side_effect = parse_side_effect
    return parser


def test_sync_session_executes_when_end_marker_seen(mock_parser: Mock):
    session = JITGenSession(
        parser=mock_parser,
        interpreter_type=MockExecutor,
        indentation_tokens={"_DEDENT", "_NEWLINE"},
        start_marker="<execute>",
        end_marker="</execute>",
    )

    assert session.push("<exe") == ""
    assert session.push("cute>print('A')\n") == ""
    assert session.push("</execute>") == "output:print('A')"

    assert session.has_executed is True
    assert session.has_output is True
    assert session.code_buffer == ""


def test_sync_session_supports_multiple_marker_windows(mock_parser: Mock):
    session = JITGenSession(
        parser=mock_parser,
        interpreter_type=MockExecutor,
        indentation_tokens={"_DEDENT", "_NEWLINE"},
        start_marker="<execute>",
        end_marker="</execute>",
    )

    output = session.push("<execute>a=1\n</execute>ignored<execute>b=2\n</execute>")

    assert "output:a=1" in output
    assert "output:b=2" in output
    assert session.code_buffer == ""


def test_sync_session_handlers_support_sync_and_async(mock_parser: Mock):
    session = JITGenSession(
        parser=mock_parser,
        interpreter_type=MockExecutor,
        indentation_tokens={"_DEDENT", "_NEWLINE"},
        start_marker="<execute>",
        end_marker="</execute>",
    )

    stdout_events: list[str] = []
    statement_events: list[tuple[str, str]] = []

    @session.on_stdout
    def capture_stdout(output: str) -> None:
        stdout_events.append(f"sync:{output}")

    @session.on_stdout
    async def capture_stdout_async(output: str) -> None:
        stdout_events.append(f"async:{output}")

    @session.on_statement_complete
    def capture_statement(statement: str, output: str) -> None:
        statement_events.append((statement.strip(), output))

    result = session.push("<execute>x=1\n</execute>")

    assert result == "output:x=1"
    assert stdout_events == ["sync:output:x=1", "async:output:x=1"]
    assert statement_events == [("x=1", "output:x=1")]


def test_sync_session_calls_error_handlers(mock_parser: Mock):
    session = JITGenSession(
        parser=mock_parser,
        interpreter_type=MockFailingExecutor,
        indentation_tokens={"_DEDENT", "_NEWLINE"},
        start_marker="<execute>",
        end_marker="</execute>",
    )

    errors: list[Exception] = []

    @session.on_error
    def capture_error(exc: Exception) -> None:
        errors.append(exc)

    with pytest.raises(ValueError, match="Error detected"):
        session.push("<execute>x=1\n</execute>")

    assert len(errors) == 1


@pytest.mark.asyncio
async def test_async_session_autoflushes_on_context_exit(mock_parser: Mock):
    stdout_events: list[str] = []

    async with AsyncJITGenSession(
        parser=mock_parser,
        interpreter_type=MockExecutor,
        indentation_tokens={"_DEDENT", "_NEWLINE"},
        start_marker="<execute>",
        end_marker="</execute>",
    ) as session:

        @session.on_stdout
        async def capture_stdout(output: str) -> None:
            stdout_events.append(output)

        await session.apush("<execute>print('hello')\n")

    assert stdout_events == ["output:print('hello')"]
