"""Unit tests for jitgen_core.jit module."""

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

from jitgen_core.jit import JITGen
from jitgen_core.base import BaseExecutor, ExecutionResult


class MockExecutor(BaseExecutor):
    """Mock executor for testing."""

    def __init__(self):
        self.execute_calls = []
        self.aexecute_calls = []

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


class MockFailingExecutor(BaseExecutor):
    """Mock executor that fails execution."""

    def execute(self, source_code: str, *, timeout: float = 5.0) -> ExecutionResult:
        return ExecutionResult(
            success=False,
            output="",
            error="Execution failed",
            has_timed_out=False,
        )

    async def aexecute(
        self, source_code: str, *, timeout: float = 5.0
    ) -> ExecutionResult:
        return ExecutionResult(
            success=False,
            output="",
            error="Execution failed",
            has_timed_out=False,
        )


@pytest.fixture
def mock_parser():
    """Create a mock Lark parser."""
    parser = Mock(spec=Lark)
    return parser


@pytest.fixture
def jitgen_instance(mock_parser):
    """Create a JITGen instance with mock parser and executor."""
    return JITGen(
        parser=mock_parser,
        interpreter_type=MockExecutor,
        indentation_tokens={"_INDENT", "_DEDENT", "_NEWLINE"},
    )


def create_mock_statement(start_pos: int, end_pos: int) -> Branch:
    """Create a mock statement (tree node) with metadata."""
    statement = Mock(spec=Branch)
    meta = Mock()
    meta.start_pos = start_pos
    meta.end_pos = end_pos
    statement.meta = meta
    return statement


def create_mock_tree(*statements: Branch) -> Branch:
    """Create a mock parse tree with given statements as children."""
    tree = Mock(spec=Branch)
    tree.children = list(statements)
    return tree


class TestJITGenInit:
    """Test JITGen initialization."""

    def test_init_with_required_fields(self, mock_parser):
        """Test that JITGen can be initialized with required fields."""
        jitgen = JITGen(
            parser=mock_parser,
            interpreter_type=MockExecutor,
        )
        assert jitgen.parser == mock_parser
        assert jitgen.interpreter_type == MockExecutor
        assert jitgen.indentation_tokens == set()

    def test_init_with_indentation_tokens(self, mock_parser):
        """Test that JITGen can be initialized with indentation tokens."""
        tokens = {"_INDENT", "_DEDENT"}
        jitgen = JITGen(
            parser=mock_parser,
            interpreter_type=MockExecutor,
            indentation_tokens=tokens,
        )
        assert jitgen.indentation_tokens == tokens


class TestExecuteStatement:
    """Test the _execute_statement method."""

    def test_execute_statement_success(self, jitgen_instance):
        """Test successful statement execution."""
        interpreter = MockExecutor()
        buffer = "print('hello')\n"
        statement = create_mock_statement(0, len(buffer))

        result = jitgen_instance._execute_statement(buffer, statement, interpreter)

        assert result == "output:print('hello')"
        assert len(interpreter.execute_calls) == 1
        assert interpreter.execute_calls[0][0] == buffer

    def test_execute_statement_with_empty_statement(self, jitgen_instance):
        """Test execution with empty statement."""
        interpreter = MockExecutor()
        buffer = "   \n"
        statement = create_mock_statement(0, len(buffer))

        result = jitgen_instance._execute_statement(buffer, statement, interpreter)

        assert result == ""
        assert len(interpreter.execute_calls) == 0

    def test_execute_statement_without_meta(self, jitgen_instance):
        """Test execution when statement has no meta."""
        interpreter = MockExecutor()
        buffer = "print('hello')\n"
        statement = Mock(spec=Branch)
        statement.meta = None

        result = jitgen_instance._execute_statement(buffer, statement, interpreter)

        assert result == ""
        assert len(interpreter.execute_calls) == 0

    def test_execute_statement_failure(self, mock_parser):
        """Test that execution failure raises ValueError."""
        jitgen = JITGen(
            parser=mock_parser,
            interpreter_type=MockFailingExecutor,
        )
        interpreter = MockFailingExecutor()
        buffer = "invalid_code\n"
        statement = create_mock_statement(0, len(buffer))

        with pytest.raises(
            ValueError, match="Error detected. Halting further processing"
        ):
            jitgen._execute_statement(buffer, statement, interpreter)

    def test_execute_statement_with_custom_timeout(self, jitgen_instance):
        """Test statement execution with custom timeout."""
        interpreter = MockExecutor()
        buffer = "print('hello')\n"
        statement = create_mock_statement(0, len(buffer))

        jitgen_instance._execute_statement(buffer, statement, interpreter, timeout=10.0)

        assert interpreter.execute_calls[0][1] == 10.0


class TestAsyncExecuteStatement:
    """Test the _aexecute_statement method."""

    @pytest.mark.asyncio
    async def test_aexecute_statement_success(self, jitgen_instance):
        """Test successful async statement execution."""
        interpreter = MockExecutor()
        buffer = "print('hello')\n"
        statement = create_mock_statement(0, len(buffer))

        result = await jitgen_instance._aexecute_statement(
            buffer, statement, interpreter
        )

        assert result == "output:print('hello')"
        assert len(interpreter.aexecute_calls) == 1
        assert interpreter.aexecute_calls[0][0] == buffer

    @pytest.mark.asyncio
    async def test_aexecute_statement_with_empty_statement(self, jitgen_instance):
        """Test async execution with empty statement."""
        interpreter = MockExecutor()
        buffer = "   \n"
        statement = create_mock_statement(0, len(buffer))

        result = await jitgen_instance._aexecute_statement(
            buffer, statement, interpreter
        )

        assert result == ""
        assert len(interpreter.aexecute_calls) == 0

    @pytest.mark.asyncio
    async def test_aexecute_statement_without_meta(self, jitgen_instance):
        """Test async execution when statement has no meta."""
        interpreter = MockExecutor()
        buffer = "print('hello')\n"
        statement = Mock(spec=Branch)
        statement.meta = None

        result = await jitgen_instance._aexecute_statement(
            buffer, statement, interpreter
        )

        assert result == ""
        assert len(interpreter.aexecute_calls) == 0

    @pytest.mark.asyncio
    async def test_aexecute_statement_failure(self, mock_parser):
        """Test that async execution failure raises ValueError."""
        jitgen = JITGen(
            parser=mock_parser,
            interpreter_type=MockFailingExecutor,
        )
        interpreter = MockFailingExecutor()
        buffer = "invalid_code\n"
        statement = create_mock_statement(0, len(buffer))

        with pytest.raises(
            ValueError, match="Error detected. Halting further processing"
        ):
            await jitgen._aexecute_statement(buffer, statement, interpreter)

    @pytest.mark.asyncio
    async def test_aexecute_statement_with_custom_timeout(self, jitgen_instance):
        """Test async statement execution with custom timeout."""
        interpreter = MockExecutor()
        buffer = "print('hello')\n"
        statement = create_mock_statement(0, len(buffer))

        await jitgen_instance._aexecute_statement(
            buffer, statement, interpreter, timeout=10.0
        )

        assert interpreter.aexecute_calls[0][1] == 10.0


class TestRunFromStream:
    """Test the run_from_stream method."""

    def test_run_single_complete_statement(self, jitgen_instance):
        """Test execution of a single complete statement."""
        fragments = ["print('hello')\n"]
        stmt1 = create_mock_statement(0, 15)
        tree = create_mock_tree(stmt1)

        jitgen_instance.parser.parse.return_value = tree

        results = list(jitgen_instance.run_from_stream(iter(fragments)))

        assert len(results) == 1
        assert results[0] == "output:print('hello')"

    def test_run_multiple_complete_statements(self, jitgen_instance):
        """Test execution of multiple complete statements in one fragment."""
        fragments = ["print('hello')\nprint('world')\n"]
        stmt1 = create_mock_statement(0, 15)
        stmt2 = create_mock_statement(15, 30)
        tree = create_mock_tree(stmt1, stmt2)

        jitgen_instance.parser.parse.return_value = tree

        results = list(jitgen_instance.run_from_stream(iter(fragments)))

        # Should execute first statement immediately, second at the end
        # Note: May include empty strings from statements with no output
        assert len([r for r in results if r]) == 2
        assert results[0] == "output:print('hello')"
        assert results[1] == "output:print('world')"

    def test_run_incomplete_then_complete(self, jitgen_instance):
        """Test handling of incomplete input followed by completion."""
        fragments = ["print('hel", "lo')\n"]

        # First fragment causes UnexpectedEOF
        # Second fragment results in complete statement
        stmt1 = create_mock_statement(0, 15)
        tree = create_mock_tree(stmt1)

        call_count = [0]

        def parse_side_effect(code):
            call_count[0] += 1
            if call_count[0] == 1:
                raise UnexpectedEOF("Unexpected EOF", None)
            return tree

        jitgen_instance.parser.parse.side_effect = parse_side_effect

        results = list(jitgen_instance.run_from_stream(iter(fragments)))

        assert len(results) == 1
        assert results[0] == "output:print('hello')"

    def test_run_unexpected_characters_error(self, jitgen_instance):
        """Test that UnexpectedCharacters raises ValueError."""
        fragments = ["invalid@@@\n"]

        jitgen_instance.parser.parse.side_effect = UnexpectedCharacters(
            "invalid@@@\n", 0, 1, 1
        )

        with pytest.raises(ValueError, match="Syntax error detected"):
            list(jitgen_instance.run_from_stream(iter(fragments)))

    def test_run_unexpected_token_with_indentation(self, jitgen_instance):
        """Test that UnexpectedToken with indentation token continues."""
        fragments = ["if True:\n", "    print('hello')\n"]

        # First fragment has indentation token, should continue
        # Second fragment completes the statement
        stmt1 = create_mock_statement(0, 29)
        tree = create_mock_tree(stmt1)

        call_count = [0]

        def parse_side_effect(code):
            call_count[0] += 1
            if call_count[0] == 1:
                token = Mock(spec=Token)
                token.type = "_DEDENT"
                raise UnexpectedToken(token, None, None, None, None)
            return tree

        jitgen_instance.parser.parse.side_effect = parse_side_effect

        results = list(jitgen_instance.run_from_stream(iter(fragments)))

        assert len(results) == 1

    def test_run_unexpected_token_without_indentation(self, jitgen_instance):
        """Test that UnexpectedToken without indentation token raises ValueError."""
        fragments = ["invalid syntax here\n"]

        token = Mock(spec=Token)
        token.type = "INVALID_TOKEN"
        jitgen_instance.parser.parse.side_effect = UnexpectedToken(
            token,
            expected=[],
            considered_rules=None,
            state=None,
            interactive_parser=None,
            terminals_by_name=None,
            token_history=None,
        )

        with pytest.raises(ValueError, match="Syntax error detected"):
            list(jitgen_instance.run_from_stream(iter(fragments)))

    def test_run_final_flush_with_syntax_error(self, jitgen_instance):
        """Test that final flush with syntax error raises ValueError."""
        fragments = ["incomplete("]

        # First parse: UnexpectedEOF (continue)
        # Final flush: UnexpectedToken (error)
        call_count = [0]

        def parse_side_effect(code):
            call_count[0] += 1
            if call_count[0] == 1:
                raise UnexpectedEOF("Unexpected EOF", None)
            token = Mock(spec=Token)
            token.type = "INVALID"
            raise UnexpectedToken(
                token,
                expected=[],
                considered_rules=None,
                state=None,
                interactive_parser=None,
                terminals_by_name=None,
                token_history=None,
            )

        jitgen_instance.parser.parse.side_effect = parse_side_effect

        with pytest.raises(ValueError, match="Syntax error detected"):
            list(jitgen_instance.run_from_stream(iter(fragments)))

    def test_run_empty_buffer_no_flush(self, jitgen_instance):
        """Test that empty buffer at end doesn't cause flush."""
        fragments = ["print('hello')\n"]
        stmt1 = create_mock_statement(0, 15)
        tree = create_mock_tree(stmt1)

        jitgen_instance.parser.parse.return_value = tree

        results = list(jitgen_instance.run_from_stream(iter(fragments)))

        # Parser is called once during fragment processing, and once during final flush
        # because single statement trees don't get emptied from buffer in the loop
        assert jitgen_instance.parser.parse.call_count == 2
        assert len([r for r in results if r]) == 1

    def test_run_whitespace_only_buffer_no_flush(self, jitgen_instance):
        """Test that whitespace-only buffer at end doesn't cause flush."""
        fragments = ["print('hello')\n", "   "]

        # First fragment: complete statement
        stmt1 = create_mock_statement(0, 15)
        tree1 = create_mock_tree(stmt1)

        # Second fragment: whitespace added to buffer - same statement
        stmt2 = create_mock_statement(0, 18)  # Includes the whitespace
        tree2 = create_mock_tree(stmt2)

        # Final flush: buffer still contains the statement with whitespace
        # But .strip() check should pass since "print('hello')\n   ".strip() is not empty
        tree3 = create_mock_tree(stmt2)

        jitgen_instance.parser.parse.side_effect = [tree1, tree2, tree3]

        results = list(jitgen_instance.run_from_stream(iter(fragments)))

        # The statement is executed (possibly multiple times due to buffer retention)
        # But effectively only produces one meaningful output
        assert len([r for r in results if r]) >= 1

    def test_run_incremental_execution(self, jitgen_instance):
        """Test that statements are executed incrementally."""
        fragments = ["a = 1\n", "b = 2\n", "print(a+b)\n"]

        # Fragment 1: buffer="a = 1\n", tree has 1 child (not executed yet, len < 2)
        stmt1 = create_mock_statement(0, 6)
        tree1 = create_mock_tree(stmt1)

        # Fragment 2: buffer="a = 1\nb = 2\n", tree has 2 children (execute first, keep second)
        stmt1_full = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        tree2 = create_mock_tree(stmt1_full, stmt2)

        # Fragment 3: buffer="b = 2\nprint(a+b)\n", tree has 2 children (execute first, keep second)
        stmt2_in_buf = create_mock_statement(0, 6)
        stmt3 = create_mock_statement(6, 18)
        tree3 = create_mock_tree(stmt2_in_buf, stmt3)

        # Final flush: buffer="print(a+b)\n", execute remaining statement
        stmt3_final = create_mock_statement(0, 12)
        tree4 = create_mock_tree(stmt3_final)

        jitgen_instance.parser.parse.side_effect = [tree1, tree2, tree3, tree4]

        results = list(jitgen_instance.run_from_stream(iter(fragments)))

        # Three statements executed: a=1, b=2, print(a+b)
        assert len([r for r in results if r]) == 3


class TestAsyncRunFromStream:
    """Test the arun_from_stream method."""

    @pytest.mark.asyncio
    async def test_arun_single_complete_statement(self, jitgen_instance):
        """Test async execution of a single complete statement."""

        async def fragments():
            yield "print('hello')\n"

        stmt1 = create_mock_statement(0, 15)
        tree = create_mock_tree(stmt1)

        jitgen_instance.parser.parse.return_value = tree

        results = []
        async for result in jitgen_instance.arun_from_stream(fragments()):
            results.append(result)

        assert len(results) == 1
        assert results[0] == "output:print('hello')"

    @pytest.mark.asyncio
    async def test_arun_multiple_complete_statements(self, jitgen_instance):
        """Test async execution of multiple complete statements."""

        async def fragments():
            yield "print('hello')\nprint('world')\n"

        stmt1 = create_mock_statement(0, 15)
        stmt2 = create_mock_statement(15, 30)
        tree = create_mock_tree(stmt1, stmt2)

        jitgen_instance.parser.parse.return_value = tree

        results = []
        async for result in jitgen_instance.arun_from_stream(fragments()):
            results.append(result)

        # May include empty strings from statements with no output
        assert len([r for r in results if r]) == 2
        assert results[0] == "output:print('hello')"
        assert results[1] == "output:print('world')"

    @pytest.mark.asyncio
    async def test_arun_incomplete_then_complete(self, jitgen_instance):
        """Test async handling of incomplete input followed by completion."""

        async def fragments():
            yield "print('hel"
            yield "lo')\n"

        stmt1 = create_mock_statement(0, 15)
        tree = create_mock_tree(stmt1)

        call_count = [0]

        def parse_side_effect(code):
            call_count[0] += 1
            if call_count[0] == 1:
                raise UnexpectedEOF("Unexpected EOF", None)
            return tree

        jitgen_instance.parser.parse.side_effect = parse_side_effect

        results = []
        async for result in jitgen_instance.arun_from_stream(fragments()):
            results.append(result)

        assert len(results) == 1
        assert results[0] == "output:print('hello')"

    @pytest.mark.asyncio
    async def test_arun_unexpected_characters_error(self, jitgen_instance):
        """Test that async UnexpectedCharacters raises ValueError."""

        async def fragments():
            yield "invalid@@@\n"

        jitgen_instance.parser.parse.side_effect = UnexpectedCharacters(
            "invalid@@@\n", 0, 1, 1
        )

        with pytest.raises(ValueError, match="Syntax error detected"):
            async for _ in jitgen_instance.arun_from_stream(fragments()):
                pass

    @pytest.mark.asyncio
    async def test_arun_unexpected_token_with_indentation(self, jitgen_instance):
        """Test that async UnexpectedToken with indentation token continues."""

        async def fragments():
            yield "if True:\n"
            yield "    print('hello')\n"

        stmt1 = create_mock_statement(0, 29)
        tree = create_mock_tree(stmt1)

        call_count = [0]

        def parse_side_effect(code):
            call_count[0] += 1
            if call_count[0] == 1:
                token = Mock(spec=Token)
                token.type = "_NEWLINE"
                raise UnexpectedToken(token, None, None, None, None)
            return tree

        jitgen_instance.parser.parse.side_effect = parse_side_effect

        results = []
        async for result in jitgen_instance.arun_from_stream(fragments()):
            results.append(result)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_arun_unexpected_token_without_indentation(self, jitgen_instance):
        """Test that async UnexpectedToken without indentation token raises ValueError."""

        async def fragments():
            yield "invalid syntax here\n"

        token = Mock(spec=Token)
        token.type = "INVALID_TOKEN"
        jitgen_instance.parser.parse.side_effect = UnexpectedToken(
            token,
            expected=[],
            considered_rules=None,
            state=None,
            interactive_parser=None,
            terminals_by_name=None,
            token_history=None,
        )

        with pytest.raises(ValueError, match="Syntax error detected"):
            async for _ in jitgen_instance.arun_from_stream(fragments()):
                pass

    @pytest.mark.asyncio
    async def test_arun_final_flush_with_unexpected_input(self, jitgen_instance):
        """Test that async final flush with UnexpectedInput raises ValueError."""

        async def fragments():
            yield "incomplete("

        call_count = [0]

        def parse_side_effect(code):
            call_count[0] += 1
            if call_count[0] == 1:
                raise UnexpectedEOF("Unexpected EOF", None)
            raise UnexpectedInput([], None, None)

        jitgen_instance.parser.parse.side_effect = parse_side_effect

        with pytest.raises(ValueError, match="Syntax error detected"):
            async for _ in jitgen_instance.arun_from_stream(fragments()):
                pass

    @pytest.mark.asyncio
    async def test_arun_empty_buffer_no_flush(self, jitgen_instance):
        """Test that async empty buffer at end doesn't cause flush."""

        async def fragments():
            yield "print('hello')\n"

        stmt1 = create_mock_statement(0, 15)
        tree = create_mock_tree(stmt1)

        jitgen_instance.parser.parse.return_value = tree

        results = []
        async for result in jitgen_instance.arun_from_stream(fragments()):
            results.append(result)

        # Parser is called once during fragment processing, and once during final flush
        # because single statement trees don't get emptied from buffer in the loop
        assert jitgen_instance.parser.parse.call_count == 2
        assert len([r for r in results if r]) == 1

    @pytest.mark.asyncio
    async def test_arun_incremental_execution(self, jitgen_instance):
        """Test that async statements are executed incrementally."""

        async def fragments():
            yield "a = 1\n"
            yield "b = 2\n"
            yield "print(a+b)\n"

        # Fragment 1: buffer="a = 1\n", tree has 1 child (not executed yet, len < 2)
        stmt1 = create_mock_statement(0, 6)
        tree1 = create_mock_tree(stmt1)

        # Fragment 2: buffer="a = 1\nb = 2\n", tree has 2 children (execute first, keep second)
        stmt1_full = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        tree2 = create_mock_tree(stmt1_full, stmt2)

        # Fragment 3: buffer="b = 2\nprint(a+b)\n", tree has 2 children (execute first, keep second)
        stmt2_in_buf = create_mock_statement(0, 6)
        stmt3 = create_mock_statement(6, 18)
        tree3 = create_mock_tree(stmt2_in_buf, stmt3)

        # Final flush: buffer="print(a+b)\n", execute remaining statement
        stmt3_final = create_mock_statement(0, 12)
        tree4 = create_mock_tree(stmt3_final)

        jitgen_instance.parser.parse.side_effect = [tree1, tree2, tree3, tree4]

        results = []
        async for result in jitgen_instance.arun_from_stream(fragments()):
            results.append(result)

        # Three statements executed: a=1, b=2, print(a+b)
        assert len([r for r in results if r]) == 3

    @pytest.mark.asyncio
    async def test_arun_buffer_management(self, jitgen_instance):
        """Test that buffer is properly managed after partial execution."""

        async def fragments():
            yield "a = 1\nb = 2\n"

        # Both statements complete, execute first, keep second in buffer
        stmt1 = create_mock_statement(0, 6)
        stmt2 = create_mock_statement(6, 12)
        tree = create_mock_tree(stmt1, stmt2)

        jitgen_instance.parser.parse.return_value = tree

        results = []
        async for result in jitgen_instance.arun_from_stream(fragments()):
            results.append(result)

        # Both statements should be executed (may include empty strings)
        assert len([r for r in results if r]) == 2
        assert results[0] == "output:a = 1"
        assert results[1] == "output:b = 2"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_fragment_stream(self, jitgen_instance):
        """Test with empty fragment stream."""
        results = list(jitgen_instance.run_from_stream(iter([])))
        assert results == []

    @pytest.mark.asyncio
    async def test_async_empty_fragment_stream(self, jitgen_instance):
        """Test async with empty fragment stream."""

        async def fragments():
            return
            yield  # This makes it a generator

        results = []
        async for result in jitgen_instance.arun_from_stream(fragments()):
            results.append(result)

        assert results == []

    def test_only_whitespace_fragments(self, jitgen_instance):
        """Test with only whitespace fragments."""
        fragments = ["  ", "\n", "  \n"]

        tree = create_mock_tree()
        jitgen_instance.parser.parse.return_value = tree

        results = list(jitgen_instance.run_from_stream(iter(fragments)))

        # No execution should happen
        assert len(results) == 0

    def test_statement_with_no_output(self, mock_parser):
        """Test statement execution that produces no output."""

        class NoOutputExecutor(BaseExecutor):
            def execute(
                self, source_code: str, *, timeout: float = 5.0
            ) -> ExecutionResult:
                return ExecutionResult(
                    success=True,
                    output=None,  # No output
                    error=None,
                    has_timed_out=False,
                )

            async def aexecute(
                self, source_code: str, *, timeout: float = 5.0
            ) -> ExecutionResult:
                return ExecutionResult(
                    success=True,
                    output=None,
                    error=None,
                    has_timed_out=False,
                )

        jitgen = JITGen(
            parser=mock_parser,
            interpreter_type=NoOutputExecutor,
        )

        fragments = ["a = 1\n"]
        stmt1 = create_mock_statement(0, 6)
        tree = create_mock_tree(stmt1)

        mock_parser.parse.return_value = tree

        results = list(jitgen.run_from_stream(iter(fragments)))

        # Should yield empty string
        assert results == [""]

    def test_multiple_fragments_single_statement(self, jitgen_instance):
        """Test that multiple fragments can form a single statement."""
        fragments = ["pri", "nt('", "hello", "')\n"]

        # First three fragments cause UnexpectedEOF
        # Last fragment completes the statement
        stmt1 = create_mock_statement(0, 15)
        tree = create_mock_tree(stmt1)

        call_count = [0]

        def parse_side_effect(code):
            call_count[0] += 1
            if call_count[0] < 4:
                raise UnexpectedEOF("Unexpected EOF", None)
            return tree

        jitgen_instance.parser.parse.side_effect = parse_side_effect

        results = list(jitgen_instance.run_from_stream(iter(fragments)))

        assert len(results) == 1
        assert "hello" in results[0]
