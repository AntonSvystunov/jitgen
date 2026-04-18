"""Tests for jitgen_langchain.parser.JITGenParser."""

import pytest

from jitgen.markers import MarkerStripper
from jitgen.prebuilt.python import create_python_jitgen
from jitgen_langchain.parser import JITGenParser


@pytest.fixture
def parser() -> JITGenParser:
    return JITGenParser(
        session=create_python_jitgen(),
        stripper=MarkerStripper(start="```python", end="```"),
    )


def test_parser_type(parser: JITGenParser):
    assert parser._type == "jitgen_parser"


def test_parse_returns_text_unchanged(parser: JITGenParser):
    assert parser.parse("anything") == "anything"


def test_transform_raises_not_implemented(parser: JITGenParser):
    with pytest.raises(NotImplementedError):
        list(parser.transform(iter(["x"])))


@pytest.mark.asyncio
async def test_atransform_single_block(parser: JITGenParser):
    async def stream():
        yield "```python\n"
        yield "print('hello')\n"
        yield "```\n"

    results = []
    async for chunk in parser._atransform(stream()):
        results.append(chunk)

    assert results == ["hello\n"]


@pytest.mark.asyncio
async def test_atransform_multiple_blocks():
    session = create_python_jitgen()
    stripper = MarkerStripper(start="```python", end="```")
    parser = JITGenParser(session=session, stripper=stripper)

    async def stream():
        yield "```python\nprint('a')\n```"
        yield " some text "
        yield "```python\nprint('b')\n```"

    results = []
    async for chunk in parser._atransform(stream()):
        results.append(chunk)

    assert results == ["a\n", "b\n"]


@pytest.mark.asyncio
async def test_atransform_runtime_error_yields_error_message(parser: JITGenParser):
    async def stream():
        yield "```python\n"
        yield "raise ValueError('oops')\n"
        yield "```\n"

    results = []
    async for chunk in parser._atransform(stream()):
        results.append(chunk)

    assert any("oops" in r or "JITGen error" in r for r in results)


@pytest.mark.asyncio
async def test_atransform_empty_block(parser: JITGenParser):
    async def stream():
        yield "```python\n"
        yield "x = 1\n"
        yield "```\n"

    results = []
    async for chunk in parser._atransform(stream()):
        results.append(chunk)

    # x = 1 produces no output
    assert results == []
