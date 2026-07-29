"""Tests for jitgen_langchain.parser.JITGenParser."""

import asyncio

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


@pytest.mark.asyncio
async def test_parser_reusable_across_streams(parser: JITGenParser):
    """The same parser must work on a second stream.

    Regression: _atransform used to aclose() the client-owned session in its
    finally, so every stream after the first silently produced nothing.
    """

    async def stream(text: str):
        yield f"```python\nprint({text!r})\n```"

    first = [chunk async for chunk in parser._atransform(stream("a"))]
    second = [chunk async for chunk in parser._atransform(stream("b"))]

    assert first == ["a\n"]
    assert second == ["b\n"]

    await parser.session.aclose()


@pytest.mark.asyncio
async def test_atransform_unterminated_block_still_executes(parser: JITGenParser):
    """A stream that ends without a closing fence must still yield its output."""

    async def stream():
        yield "```python\n"
        yield "print('tail')\n"

    results = [chunk async for chunk in parser._atransform(stream())]

    assert results == ["tail\n"]
    # State must be clean for the next stream despite the missing end marker.
    assert not parser.stripper.inside_markers
    assert not parser.session.has_error

    await parser.session.aclose()


@pytest.mark.asyncio
async def test_parser_usable_after_mid_stream_cancellation(parser: JITGenParser):
    """Cancelling a stream must not corrupt the session for the next one."""

    async def slow_stream():
        yield "```python\n"
        yield "print('lost')\n"
        await asyncio.sleep(10)
        yield "```"

    async def consume(agen):
        return [chunk async for chunk in agen]

    task = asyncio.create_task(consume(parser._atransform(slow_stream())))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async def stream():
        yield "```python\nprint('after')\n```"

    results = [chunk async for chunk in parser._atransform(stream())]
    assert results == ["after\n"]

    await parser.session.aclose()
