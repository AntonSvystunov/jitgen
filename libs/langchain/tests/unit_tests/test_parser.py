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


# ── incremental forwarding ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_output_forwarded_before_the_closing_marker(parser: JITGenParser):
    """stdout must not be banked until the end marker arrives.

    Holding it back pins time-to-first-output to the end of generation, which is
    exactly what the sequential baseline does — the incremental path would show no
    first-output benefit at all.  ``sent`` records what the model has produced so
    far, so a chunk arriving while the fence is still unsent proves the output was
    forwarded mid-stream rather than at the final flush.
    """
    sent: list[str] = []

    async def stream():
        for text in ["```python\n", "print('early')\n", "x = 1\n", "y = 2\n", "```"]:
            sent.append(text)
            yield text
            await asyncio.sleep(0.01)  # a real stream awaits between chunks

    fence_sent_at_first_output = None
    async for chunk in parser._atransform(stream()):
        if fence_sent_at_first_output is None:
            fence_sent_at_first_output = "```" in sent[1:]
            assert chunk == "early\n"

    assert fence_sent_at_first_output is False, (
        "first output arrived only after the closing fence — not incremental"
    )


@pytest.mark.asyncio
async def test_incremental_forwarding_preserves_order_and_content(
    parser: JITGenParser,
):
    async def stream():
        for char in "```python\nprint(1)\nprint(2)\nprint(3)\n```":
            yield char  # one character per chunk: worst-case boundaries
            await asyncio.sleep(0)

    chunks = [c async for c in parser._atransform(stream())]
    assert "".join(chunks) == "1\n2\n3\n"
    assert len(chunks) > 1, "output arrived in one lump — nothing was streamed"


@pytest.mark.asyncio
async def test_no_output_duplicated_between_drain_and_final_flush(
    parser: JITGenParser,
):
    async def stream():
        yield "```python\n"
        yield "print('a')\n"
        await asyncio.sleep(0.01)  # let the worker run, so a drain happens
        yield "print('b')\n"
        await asyncio.sleep(0.01)
        yield "```"

    chunks = [c async for c in parser._atransform(stream())]
    assert "".join(chunks) == "a\nb\n"
