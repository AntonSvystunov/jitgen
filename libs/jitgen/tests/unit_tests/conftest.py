import asyncio
from collections.abc import AsyncIterator

import pytest
from lark import Lark
from lark.indenter import PythonIndenter

from jitgen import ExecutionResult, PythonLarkExtractor, Session
from jitgen.executors.base import ExecutorBase


class FakeExtractor:
    r"""Newline-delimited `StatementExtractor` double.

    Each `\n`-terminated line becomes one statement; a trailing partial line
    is held back until `final=True` or later input completes it. A line equal
    to `SYNTAX_ERROR_LINE` raises `SyntaxError`, so tests can exercise
    `Session`'s extraction-failure path without a real grammar.
    """

    SYNTAX_ERROR_LINE = "<<SYNTAX ERROR>>"

    def extract(self, source: str, *, final: bool) -> tuple[list[str], str]:
        lines = source.split("\n")
        if final:
            complete, leftover = lines, ""
        else:
            *complete, leftover = lines
        statements: list[str] = []
        for line in complete:
            if not line:
                continue
            if line == self.SYNTAX_ERROR_LINE:
                raise SyntaxError(f"unrecoverable statement: {line!r}")
            statements.append(line)
        return statements, leftover


class FakeExecutor(ExecutorBase):
    """Scriptable `BaseExecutor` double with call recording and gating.

    By default every statement succeeds with no output. Use `respond` or
    `raise_for` to script a specific outcome, and `gate`/`started` to control
    and observe timing across the background worker.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.acancel_calls = 0
        self.aclose_calls = 0
        self.raise_on_acancel: BaseException | None = None
        self._responses: dict[str, ExecutionResult | BaseException] = {}
        self._gates: dict[str, asyncio.Event] = {}
        self._started: dict[str, asyncio.Event] = {}

    def respond(
        self,
        statement: str,
        *,
        output: str = "",
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Configure the `ExecutionResult` returned for `statement`."""
        self._responses[statement] = ExecutionResult(
            success=success, output=output or None, error=error
        )

    def raise_for(self, statement: str, exc: BaseException) -> None:
        """Configure `aexecute(statement)` to raise `exc` instead of returning."""
        self._responses[statement] = exc

    def gate(self, statement: str) -> asyncio.Event:
        """Return an event; `aexecute(statement)` blocks until it is set.

        `acancel()` sets every outstanding gate, simulating an executor that
        actually honors cancellation of in-flight work.
        """
        event = asyncio.Event()
        self._gates[statement] = event
        return event

    def started(self, statement: str) -> asyncio.Event:
        """Return the event set the moment `aexecute(statement)` begins running."""
        return self._started.setdefault(statement, asyncio.Event())

    async def aexecute(self, source_code: str) -> ExecutionResult:
        self.calls.append(source_code)
        self.started(source_code).set()
        gate = self._gates.get(source_code)
        if gate is not None:
            await gate.wait()
        outcome = self._responses.get(source_code, ExecutionResult(success=True))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def acancel(self) -> None:
        self.acancel_calls += 1
        if self.raise_on_acancel is not None:
            raise self.raise_on_acancel
        for event in self._gates.values():
            event.set()

    async def aclose(self) -> None:
        self.aclose_calls += 1


@pytest.fixture
def extractor() -> FakeExtractor:
    return FakeExtractor()


@pytest.fixture
def executor() -> FakeExecutor:
    return FakeExecutor()


@pytest.fixture
async def session(
    extractor: FakeExtractor, executor: FakeExecutor
) -> AsyncIterator[Session]:
    instance = Session(extractor=extractor, executor=executor, owns_executor=True)
    try:
        yield instance
    finally:
        await instance.aclose()


@pytest.fixture(scope="session")
def python_lark_parser() -> Lark:
    # Session-scoped: building the LALR tables takes ~0.4s and the parser
    # holds no per-parse state, so every test can safely share one instance.
    return Lark.open_from_package(
        "lark",
        "python.lark",
        ["grammars"],
        parser="lalr",
        postlex=PythonIndenter(),
        start="file_input",
        maybe_placeholders=False,
        propagate_positions=True,
    )


@pytest.fixture
def python_extractor(python_lark_parser: Lark) -> PythonLarkExtractor:
    return PythonLarkExtractor(python_lark_parser)
