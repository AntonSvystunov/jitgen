from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from jitgen_js.executors.quickjs import QuickJsExecutor
from jitgen_js.extractors.javascript import JavaScriptAntlrExtractor


@pytest.fixture
def extractor() -> JavaScriptAntlrExtractor:
    # No session-scoping needed, unlike Lark's `python_lark_parser` fixture:
    # ANTLR's generated Lexer/Parser classes are cheap to construct (the
    # expensive table-building step already happened once, ahead of time,
    # when the grammar was vendored) — there is no LALR-table cost to
    # amortize across tests.
    return JavaScriptAntlrExtractor()


@pytest.fixture
async def make_executor() -> AsyncIterator[Callable[..., QuickJsExecutor]]:
    # Factory rather than a fixed instance: several tests need their own
    # timeout/tools. Every instance created through it is closed on teardown
    # so no worker thread outlives its test.
    created: list[QuickJsExecutor] = []

    def factory(**kwargs: Any) -> QuickJsExecutor:
        instance = QuickJsExecutor(**kwargs)
        created.append(instance)
        return instance

    try:
        yield factory
    finally:
        for instance in created:
            await instance.aclose()


@pytest.fixture
def executor(make_executor: Callable[..., QuickJsExecutor]) -> QuickJsExecutor:
    return make_executor()
