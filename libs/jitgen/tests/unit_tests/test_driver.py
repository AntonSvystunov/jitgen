import asyncio

import pytest

from jitgen import ExecutionError, ExtractionError, Session, StreamDriver
from jitgen.segmenters import markdown_code

from .conftest import FakeExecutor, FakeExtractor


@pytest.fixture
def driver(session: Session) -> StreamDriver:
    return StreamDriver(session, markdown_code("python"))


async def test_apush_discards_text_outside_a_block(
    driver: StreamDriver, session: Session
):
    out = await driver.apush("some prose before the fence, no fence at all")

    assert out == []
    assert session.buffer == ""


async def test_apush_returns_output_immediately_when_the_block_closes_in_one_chunk(
    driver: StreamDriver, executor: FakeExecutor
):
    executor.respond("a", output="A")
    executor.respond("b", output="B")

    out = await driver.apush("```python\na\nb\n```")

    assert out == ["AB"]
    assert executor.calls == ["a", "b"]


async def test_apush_accumulates_output_across_chunks_until_the_block_closes(
    driver: StreamDriver, executor: FakeExecutor
):
    executor.respond("a", output="A")
    executor.respond("b", output="B")

    first = await driver.apush("```python\na\n")
    second = await driver.apush("b\n```")

    # Exactly when "a"'s output surfaces relative to "b"'s isn't the contract
    # here (that's covered separately below); only that each part of the
    # total output is returned exactly once, in order.
    assert "".join(first) + "".join(second) == "AB"


async def test_apush_surfaces_output_from_an_earlier_statement_while_the_block_is_still_open(
    driver: StreamDriver, executor: FakeExecutor
):
    executor.respond("a", output="A")

    await driver.apush("```python\na\n")
    # Give the background worker a chance to actually run "a" before the next
    # chunk arrives -- this is the mechanism `driver.py`'s "drained per chunk"
    # comment describes: output surfaces without waiting for the closing fence.
    await asyncio.sleep(0)

    assert await driver.apush("b") == ["A"]


async def test_apush_raises_immediately_when_a_closing_block_fails_to_execute(
    driver: StreamDriver, executor: FakeExecutor
):
    executor.raise_for("a", ValueError("boom"))

    with pytest.raises(ExecutionError):
        await driver.apush("```python\na\n```")


async def test_apush_does_not_raise_for_an_error_detected_mid_block(
    driver: StreamDriver, extractor: FakeExtractor
):
    out = await driver.apush(f"```python\n{extractor.SYNTAX_ERROR_LINE}\n")

    assert out == []
    assert driver.has_error
    assert isinstance(driver.error, ExtractionError)


async def test_apush_does_not_raise_even_when_the_failing_blocks_own_fence_closes(
    driver: StreamDriver, extractor: FakeExtractor
):
    # A single chunk can carry a whole failing block *and* a second, healthy
    # one after it (e.g. two fenced snippets in one streamed message chunk).
    # The failing block's own fence closes cleanly in this same chunk, which
    # is exactly the case `apush`'s "has_error" guard must catch before ever
    # reaching `session.result()` for it -- otherwise the extraction failure
    # would surface as a raise from `apush` instead of being left on the
    # session, and the second block would never even be looked at.
    chunk = f"```python\n{extractor.SYNTAX_ERROR_LINE}\n```\n```python\nx\n```"

    out = await driver.apush(chunk)

    assert out == []
    assert driver.has_error


async def test_afinish_reraises_an_error_recorded_earlier_in_the_stream(
    driver: StreamDriver, extractor: FakeExtractor
):
    await driver.apush(f"```python\n{extractor.SYNTAX_ERROR_LINE}\n")

    with pytest.raises(ExtractionError):
        await driver.afinish()


async def test_afinish_executes_a_block_the_model_never_closed(
    driver: StreamDriver, executor: FakeExecutor
):
    executor.respond("a", output="A")
    executor.respond("b", output="B")

    # No closing fence, and "b" has no trailing newline: the extractor holds
    # it back as the still-growing last statement (P2) until the flush.
    await driver.apush("```python\na\nb")

    assert await driver.afinish() == ["AB"]
    assert executor.calls == ["a", "b"]


async def test_afinish_flushes_a_segmenter_suffix_that_looked_like_a_marker(
    driver: StreamDriver, executor: FakeExecutor
):
    executor.respond("a", output="A")
    executor.respond("``", output="X")

    # The stream ends with two backticks -- a plausible prefix of the closing
    # fence -- so the segmenter withholds them until it learns no third
    # backtick is ever coming, only releasing them from `finalize()`.
    await driver.apush("```python\na\n``")

    assert await driver.afinish() == ["AX"]


async def test_afinish_returns_empty_list_when_the_stream_produced_no_output(
    driver: StreamDriver,
):
    await driver.apush("```python\n```")

    assert await driver.afinish() == []


async def test_areset_clears_segmenter_and_session_state_but_keeps_the_executor(
    driver: StreamDriver, session: Session, executor: FakeExecutor
):
    await driver.apush("```python\na\n")  # unterminated block

    await driver.areset()

    assert session.buffer == ""
    assert not driver.session.has_error
    assert executor.aclose_calls == 0  # REPL state / executor is preserved

    # The segmenter was reset too: raw text is once again "outside a block".
    out = await driver.apush("more prose, still no fence")
    assert out == []


async def test_has_error_and_error_delegate_to_the_session(
    driver: StreamDriver, session: Session, extractor: FakeExtractor
):
    assert driver.has_error is False
    assert driver.error is None

    await driver.apush(f"```python\n{extractor.SYNTAX_ERROR_LINE}\n")

    assert driver.has_error is True
    assert driver.error is session.error


def test_session_property_exposes_the_underlying_session(
    driver: StreamDriver, session: Session
):
    assert driver.session is session
