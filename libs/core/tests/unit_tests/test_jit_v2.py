"""Unit tests for jitgen_core.v2.jit module (JITGenV2 – push-based API)."""

import pytest
from unittest.mock import Mock
from lark import (
    Lark,
    UnexpectedEOF,
    UnexpectedToken,
    UnexpectedCharacters,
    UnexpectedInput,
    Token,
)
from lark.tree import Branch

from jitgen_core.v2.jit import JITGenV2
from jitgen_core.base import BaseExecutor, ExecutionResult


# ── test doubles ───────────────────────────────────────────────────────


class MockExecutor(BaseExecutor):
    """Mock executor that echoes the source code as output."""

    def __init__(self):
        self.execute_calls: list[tuple[str, float]] = []
        self.aexecute_calls: list[tuple[str, float]] = []

    def execute(self, source_code: str, *, timeout: float = 5.0) -> ExecutionResult:
        self.execute_calls.append((source_code, timeout))
        return ExecutionResult(
            success=True,
            output=f"output:{source_code.strip()}",
            error=None,
            has_timed_out=False,
        )

    async def aexecute(
        self, source_code: str, *, timeout: float = 5.0
    ) -> ExecutionResult:
        self.aexecute_calls.append((source_code, timeout))
        return ExecutionResult(
            success=True,
            output=f"output:{source_code.strip()}",
            error=None,
            has_timed_out=False,
        )


class MockSilentExecutor(BaseExecutor):
    """Mock executor that succeeds but produces no output."""

    def __init__(self):
        self.execute_calls: list[tuple[str, float]] = []
        self.aexecute_calls: list[tuple[str, float]] = []

    def execute(self, source_code: str, *, timeout: float = 5.0) -> ExecutionResult:
        self.execute_calls.append((source_code, timeout))
        return ExecutionResult(success=True, output="", error=None)

    async def aexecute(
        self, source_code: str, *, timeout: float = 5.0
    ) -> ExecutionResult:
        self.aexecute_calls.append((source_code, timeout))
        return ExecutionResult(success=True, output="", error=None)


class MockFailingExecutor(BaseExecutor):
    """Mock executor that always fails."""

    def __init__(self): ...

    def execute(self, source_code: str, *, timeout: float = 5.0) -> ExecutionResult:
        return ExecutionResult(
            success=False, output="", error="Execution failed", has_timed_out=False
        )

    async def aexecute(
        self, source_code: str, *, timeout: float = 5.0
    ) -> ExecutionResult:
        return ExecutionResult(
            success=False, output="", error="Execution failed", has_timed_out=False
        )


# ── helpers ────────────────────────────────────────────────────────────


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


# ── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def mock_parser():
    return Mock(spec=Lark)


@pytest.fixture
def v2(mock_parser):
    """Create a JITGenV2 instance backed by a mock parser and MockExecutor."""
    return JITGenV2(
        parser=mock_parser,
        interpreter_type=MockExecutor,
        indentation_tokens={"_INDENT", "_DEDENT", "_NEWLINE"},
    )


# ── TestJITGenV2Init ──────────────────────────────────────────────────


class TestJITGenV2Init:
    """Initialization and default state."""

    def test_required_fields(self, mock_parser):
        inst = JITGenV2(parser=mock_parser, interpreter_type=MockExecutor)
        assert inst.parser is mock_parser
        assert inst.interpreter_type is MockExecutor
        assert inst.indentation_tokens == set()

    def test_indentation_tokens(self, mock_parser):
        tokens = {"_INDENT", "_DEDENT"}
        inst = JITGenV2(
            parser=mock_parser,
            interpreter_type=MockExecutor,
            indentation_tokens=tokens,
        )
        assert inst.indentation_tokens == tokens

    def test_initial_state(self, v2):
        assert v2.has_executed is False
        assert v2.has_output is False
        assert v2.chunks == []
        assert v2.code_buffer == ""

    def test_interpreter_created_eagerly(self, mock_parser):
        inst = JITGenV2(parser=mock_parser, interpreter_type=MockExecutor)
        assert isinstance(inst._interpreter, MockExecutor)


# ── TestExecuteStatement ──────────────────────────────────────────────


class TestExecuteStatement:
    """Low-level _execute_statement tests."""

    def test_success(self, v2):
        buffer = "print('hello')\n"
        stmt = create_mock_statement(0, len(buffer))
        result = v2._execute_statement(buffer, stmt)
        assert result == "output:print('hello')"
        assert v2.has_executed is True
        assert v2.has_output is True

    def test_empty_statement(self, v2):
        buffer = "   \n"
        stmt = create_mock_statement(0, len(buffer))
        result = v2._execute_statement(buffer, stmt)
        assert result == ""
        assert v2.has_executed is False

    def test_no_meta(self, v2):
        buffer = "print('hello')\n"
        stmt = Mock(spec=Branch)
        stmt.meta = None
        result = v2._execute_statement(buffer, stmt)
        assert result == ""

    def test_failure_raises(self, mock_parser):
        inst = JITGenV2(
            parser=mock_parser, interpreter_type=MockFailingExecutor
        )
        buffer = "bad_code\n"
        stmt = create_mock_statement(0, len(buffer))
        with pytest.raises(ValueError, match="Error detected"):
            inst._execute_statement(buffer, stmt)

    def test_custom_timeout(self, v2):
        buffer = "x = 1\n"
        stmt = create_mock_statement(0, len(buffer))
        v2._execute_statement(buffer, stmt, timeout=12.0)
        assert v2._interpreter.execute_calls[0][1] == 12.0


# ── TestAsyncExecuteStatement ─────────────────────────────────────────


class TestAsyncExecuteStatement:
    @pytest.mark.asyncio
    async def test_success(self, v2):
        buffer = "print('hello')\n"
        stmt = create_mock_statement(0, len(buffer))
        result = await v2._aexecute_statement(buffer, stmt)
        assert result == "output:print('hello')"
        assert v2.has_executed is True
        assert v2.has_output is True

    @pytest.mark.asyncio
    async def test_empty_statement(self, v2):
        buffer = "   \n"
        stmt = create_mock_statement(0, len(buffer))
        result = await v2._aexecute_statement(buffer, stmt)
        assert result == ""
        assert v2.has_executed is False

    @pytest.mark.asyncio
    async def test_no_meta(self, v2):
        buffer = "x = 1\n"
        stmt = Mock(spec=Branch)
        stmt.meta = None
        result = await v2._aexecute_statement(buffer, stmt)
        assert result == ""

    @pytest.mark.asyncio
    async def test_failure_raises(self, mock_parser):
        inst = JITGenV2(
            parser=mock_parser, interpreter_type=MockFailingExecutor
        )
        buffer = "bad\n"
        stmt = create_mock_statement(0, len(buffer))
        with pytest.raises(ValueError, match="Error detected"):
            await inst._aexecute_statement(buffer, stmt)


# ── TestPush ──────────────────────────────────────────────────────────


class TestPush:
    """Synchronous push() tests."""

    def test_single_complete_statement(self, v2, mock_parser):
        """Two children → first gets executed, second remains in buffer."""
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2)

        result = v2.push("x = 1\ny = 2\n")

        assert result == "output:x = 1"
        assert v2.chunks == ["x = 1\ny = 2\n"]
        assert v2.code_buffer == "y = 2\n"
        assert v2.has_executed is True
        assert v2.has_output is True

    def test_incomplete_input_unexpected_eof(self, v2, mock_parser):
        mock_parser.parse.side_effect = UnexpectedEOF([])
        result = v2.push("if True:")
        assert result == ""
        assert v2.code_buffer == "if True:"
        assert v2.has_executed is False

    def test_incomplete_input_indent_token(self, v2, mock_parser):
        token = Mock()
        token.type = "_DEDENT"
        mock_parser.parse.side_effect = UnexpectedToken(token, [])
        result = v2.push("def f():\n  pass")
        assert result == ""

    def test_syntax_error_unexpected_characters(self, v2, mock_parser):
        mock_parser.parse.side_effect = UnexpectedCharacters("@", 0, 1, 1)
        with pytest.raises(ValueError, match="Syntax error detected"):
            v2.push("@@@")

    def test_syntax_error_unexpected_token(self, v2, mock_parser):
        token = Mock()
        token.type = "NAME"  # not an indentation token
        mock_parser.parse.side_effect = UnexpectedToken(token, [])
        with pytest.raises(ValueError, match="Syntax error detected"):
            v2.push("def 123")

    def test_only_one_child_no_execution(self, v2, mock_parser):
        """A single child means we can't be sure it's complete."""
        stmt = create_mock_statement(0, 6)
        mock_parser.parse.return_value = create_mock_tree(stmt)
        result = v2.push("x = 1\n")
        assert result == ""
        assert v2.has_executed is False

    def test_accumulates_chunks(self, v2, mock_parser):
        mock_parser.parse.side_effect = UnexpectedEOF([])
        v2.push("a")
        v2.push("b")
        v2.push("c")
        assert v2.chunks == ["a", "b", "c"]
        assert v2.code_buffer == "abc"

    def test_multiple_statements_executed(self, v2, mock_parser):
        """Three children → first two executed, third remains."""
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        stmt3 = create_mock_statement(12, 18)
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2, stmt3)

        result = v2.push("a = 1\nb = 2\nc = 3\n")

        assert "output:a = 1" in result
        assert "output:b = 2" in result
        assert v2.code_buffer == "c = 3\n"

    def test_custom_timeout_forwarded(self, v2, mock_parser):
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2)

        v2.push("x = 1\ny = 2\n", timeout=15.0)
        assert v2._interpreter.execute_calls[0][1] == 15.0

    def test_whitespace_only_no_parse(self, v2, mock_parser):
        result = v2.push("   \n  ")
        mock_parser.parse.assert_not_called()
        assert result == ""

    def test_execution_error_raises(self, mock_parser):
        inst = JITGenV2(
            parser=mock_parser, interpreter_type=MockFailingExecutor
        )
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2)
        with pytest.raises(ValueError, match="Error detected"):
            inst.push("bad()\nmore\n")


# ── TestApush ─────────────────────────────────────────────────────────


class TestApush:
    """Async apush() tests — mirrors TestPush."""

    @pytest.mark.asyncio
    async def test_single_complete_statement(self, v2, mock_parser):
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2)

        result = await v2.apush("x = 1\ny = 2\n")

        assert result == "output:x = 1"
        assert v2.chunks == ["x = 1\ny = 2\n"]
        assert v2.code_buffer == "y = 2\n"
        assert v2.has_executed is True

    @pytest.mark.asyncio
    async def test_incomplete_input(self, v2, mock_parser):
        mock_parser.parse.side_effect = UnexpectedEOF([])
        result = await v2.apush("if True:")
        assert result == ""
        assert v2.has_executed is False

    @pytest.mark.asyncio
    async def test_indent_token_incomplete(self, v2, mock_parser):
        token = Mock()
        token.type = "_DEDENT"
        mock_parser.parse.side_effect = UnexpectedToken(token, [])
        result = await v2.apush("def f():\n  pass")
        assert result == ""

    @pytest.mark.asyncio
    async def test_syntax_error(self, v2, mock_parser):
        mock_parser.parse.side_effect = UnexpectedCharacters("@", 0, 1, 1)
        with pytest.raises(ValueError, match="Syntax error detected"):
            await v2.apush("@@@")

    @pytest.mark.asyncio
    async def test_accumulates_chunks(self, v2, mock_parser):
        mock_parser.parse.side_effect = UnexpectedEOF([])
        await v2.apush("a")
        await v2.apush("b")
        assert v2.chunks == ["a", "b"]
        assert v2.code_buffer == "ab"

    @pytest.mark.asyncio
    async def test_multiple_statements(self, v2, mock_parser):
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        stmt3 = create_mock_statement(12, 18)
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2, stmt3)

        result = await v2.apush("a = 1\nb = 2\nc = 3\n")
        assert "output:a = 1" in result
        assert "output:b = 2" in result


# ── TestFlush ─────────────────────────────────────────────────────────


class TestFlush:
    """Synchronous flush() tests."""

    def test_flush_remaining_buffer(self, v2, mock_parser):
        """Flush executes ALL children (including the last one)."""
        # First push some incomplete code
        mock_parser.parse.side_effect = UnexpectedEOF([])
        v2.push("x = 1\n")

        # Now set up parse for flush
        stmt = create_mock_statement(0, 6)
        mock_parser.parse.side_effect = None
        mock_parser.parse.return_value = create_mock_tree(stmt)

        result = v2.flush()
        assert result == "output:x = 1"
        assert v2.code_buffer == ""
        assert v2.has_executed is True

    def test_flush_empty_buffer(self, v2):
        result = v2.flush()
        assert result == ""

    def test_flush_whitespace_only(self, v2, mock_parser):
        mock_parser.parse.side_effect = UnexpectedEOF([])
        v2.push("   \n  ")
        # buffer is whitespace from push, but re-check in flush:
        # _process_buffer(flush=True) with whitespace-only should return ""
        result = v2.flush()
        assert result == ""

    def test_flush_syntax_error_eof(self, v2, mock_parser):
        """UnexpectedEOF during flush raises ValueError."""
        # Prime the buffer
        v2._code_buffer = "if True:"
        mock_parser.parse.side_effect = UnexpectedEOF([])
        with pytest.raises(ValueError, match="Syntax error detected"):
            v2.flush()

    def test_flush_syntax_error_token(self, v2, mock_parser):
        """UnexpectedToken during flush raises (even for indent tokens)."""
        v2._code_buffer = "def f():"
        token = Mock()
        token.type = "_DEDENT"  # indent token — in push this is tolerated, in flush it's an error
        mock_parser.parse.side_effect = UnexpectedToken(token, [])
        with pytest.raises(ValueError, match="Syntax error detected"):
            v2.flush()

    def test_flush_multiple_children(self, v2, mock_parser):
        v2._code_buffer = "a = 1\nb = 2\n"
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2)

        result = v2.flush()
        assert "output:a = 1" in result
        assert "output:b = 2" in result
        assert v2.code_buffer == ""

    def test_flush_custom_timeout(self, v2, mock_parser):
        v2._code_buffer = "x = 1\n"
        stmt = create_mock_statement(0, 6)
        mock_parser.parse.return_value = create_mock_tree(stmt)
        v2.flush(timeout=20.0)
        assert v2._interpreter.execute_calls[0][1] == 20.0


# ── TestAflush ────────────────────────────────────────────────────────


class TestAflush:
    """Async aflush() tests — mirrors TestFlush."""

    @pytest.mark.asyncio
    async def test_flush_remaining(self, v2, mock_parser):
        mock_parser.parse.side_effect = UnexpectedEOF([])
        await v2.apush("x = 1\n")

        stmt = create_mock_statement(0, 6)
        mock_parser.parse.side_effect = None
        mock_parser.parse.return_value = create_mock_tree(stmt)

        result = await v2.aflush()
        assert result == "output:x = 1"
        assert v2.code_buffer == ""

    @pytest.mark.asyncio
    async def test_flush_empty(self, v2):
        result = await v2.aflush()
        assert result == ""

    @pytest.mark.asyncio
    async def test_flush_syntax_error(self, v2, mock_parser):
        v2._code_buffer = "if True:"
        mock_parser.parse.side_effect = UnexpectedEOF([])
        with pytest.raises(ValueError, match="Syntax error detected"):
            await v2.aflush()

    @pytest.mark.asyncio
    async def test_flush_multiple_children(self, v2, mock_parser):
        v2._code_buffer = "a = 1\nb = 2\n"
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2)

        result = await v2.aflush()
        assert "output:a = 1" in result
        assert "output:b = 2" in result
        assert v2.code_buffer == ""


# ── TestStateTracking ─────────────────────────────────────────────────


class TestStateTracking:
    """Verify has_executed / has_output / chunks tracking."""

    def test_has_executed_false_when_only_buffering(self, v2, mock_parser):
        mock_parser.parse.side_effect = UnexpectedEOF([])
        v2.push("x = 1")
        assert v2.has_executed is False

    def test_has_executed_true_after_execution(self, v2, mock_parser):
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2)
        v2.push("x = 1\ny = 2\n")
        assert v2.has_executed is True

    def test_has_output_false_with_silent_executor(self, mock_parser):
        inst = JITGenV2(
            parser=mock_parser,
            interpreter_type=MockSilentExecutor,
            indentation_tokens={"_INDENT", "_DEDENT", "_NEWLINE"},
        )
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2)
        inst.push("x = 1\ny = 2\n")
        assert inst.has_executed is True
        assert inst.has_output is False

    def test_has_output_true_with_output(self, v2, mock_parser):
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2)
        v2.push("x = 1\ny = 2\n")
        assert v2.has_output is True

    def test_chunks_returns_defensive_copy(self, v2, mock_parser):
        mock_parser.parse.side_effect = UnexpectedEOF([])
        v2.push("a")
        chunks = v2.chunks
        chunks.append("mutated")
        assert v2.chunks == ["a"]

    def test_chunks_accumulates(self, v2, mock_parser):
        mock_parser.parse.side_effect = UnexpectedEOF([])
        v2.push("chunk1")
        v2.push("chunk2")
        v2.push("chunk3")
        assert v2.chunks == ["chunk1", "chunk2", "chunk3"]

    def test_code_buffer_tracks_unparsed(self, v2, mock_parser):
        """After executing first statement, buffer holds only the remainder."""
        # First push — incomplete
        mock_parser.parse.side_effect = UnexpectedEOF([])
        v2.push("x = 1\n")
        assert v2.code_buffer == "x = 1\n"

        # Second push — two complete children
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        mock_parser.parse.side_effect = None
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2)
        v2.push("y = 2\n")
        assert v2.code_buffer == "y = 2\n"


# ── TestEdgeCases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_chunk(self, v2, mock_parser):
        result = v2.push("")
        assert result == ""
        assert v2.chunks == [""]

    def test_whitespace_only_chunks(self, v2, mock_parser):
        result = v2.push("   ")
        assert result == ""
        result = v2.push("\n\n")
        assert result == ""

    def test_push_then_flush_full_workflow(self, v2, mock_parser):
        """Integration: push several chunks, then flush."""
        # Chunk 1 — incomplete
        mock_parser.parse.side_effect = UnexpectedEOF([])
        r1 = v2.push("x = 1\n")
        assert r1 == ""

        # Chunk 2 — two complete children
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        tree_2 = create_mock_tree(stmt1, stmt2)
        mock_parser.parse.side_effect = None
        mock_parser.parse.return_value = tree_2
        r2 = v2.push("y = 2\n")
        assert "output:" in r2

        # Flush the remainder
        stmt_last = create_mock_statement(0, 6)
        mock_parser.parse.return_value = create_mock_tree(stmt_last)
        r3 = v2.flush()
        assert "output:" in r3
        assert v2.code_buffer == ""
        assert v2.has_executed is True
        assert len(v2.chunks) == 2

    @pytest.mark.asyncio
    async def test_apush_then_aflush_full_workflow(self, v2, mock_parser):
        """Async integration: apush + aflush."""
        mock_parser.parse.side_effect = UnexpectedEOF([])
        r1 = await v2.apush("x = 1\n")
        assert r1 == ""

        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        mock_parser.parse.side_effect = None
        mock_parser.parse.return_value = create_mock_tree(stmt1, stmt2)
        r2 = await v2.apush("y = 2\n")
        assert "output:" in r2

        stmt_last = create_mock_statement(0, 6)
        mock_parser.parse.return_value = create_mock_tree(stmt_last)
        r3 = await v2.aflush()
        assert "output:" in r3
        assert v2.code_buffer == ""

    def test_flush_after_no_pushes(self, v2):
        """Flushing without ever pushing should be a no-op."""
        result = v2.flush()
        assert result == ""
        assert v2.has_executed is False

    def test_execution_failure_during_flush(self, mock_parser):
        inst = JITGenV2(
            parser=mock_parser, interpreter_type=MockFailingExecutor
        )
        inst._code_buffer = "bad()\n"
        stmt = create_mock_statement(0, 6)
        mock_parser.parse.return_value = create_mock_tree(stmt)
        with pytest.raises(ValueError, match="Error detected"):
            inst.flush()
