import asyncio

import pytest
from conftest import (
    FakeChunk,
    FakeClient,
    FakeCompletions,
    FakeStream,
    FakeUsage,
    FakeUsageDetails,
    content_chunk,
    reasoning_chunk,
    tool_chunk,
)

from iptc_bfcl.instrumentation import (
    InstrumentedClient,
    count_tokens,
    truncate_tool_results,
)
from iptc_bfcl.metrics import RunRecorder


async def test_completed_stream_records_timings_usage_and_estimates():
    usage = FakeChunk(usage=FakeUsage(120, 30, FakeUsageDetails(10)))
    fake = FakeClient(
        [[reasoning_chunk("think "), content_chunk("hello", finish="stop"), usage]]
    )
    recorder = RunRecorder()
    client = InstrumentedClient(fake, recorder, {"stream_options": {"x": True}})

    stream = await client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}], stream=True
    )
    async with stream:
        chunks = [chunk async for chunk in stream]

    assert len(chunks) == 3
    assert fake.chat.completions.calls[0]["stream_options"] == {"x": True}
    [turn] = recorder.turns
    assert turn.completed
    assert turn.finish_reason == "stop"
    assert turn.generation_id == "gen-1"
    assert (turn.usage_input, turn.usage_output, turn.usage_reasoning) == (120, 30, 10)
    assert turn.reasoning_text == "think "
    assert turn.content_text == "hello"
    assert turn.est_reasoning == count_tokens("think ")
    assert turn.est_output == count_tokens("think ") + count_tokens("hello")
    assert turn.est_input > 0
    assert turn.first_reasoning <= turn.first_content <= turn.end
    assert fake.chat.completions.streams[0].closed


async def test_stream_closed_early_is_not_completed():
    fake = FakeClient([[tool_chunk('{"code": "a'), tool_chunk("b"), tool_chunk('"}')]])
    recorder = RunRecorder()
    client = InstrumentedClient(fake, recorder)

    stream = await client.chat.completions.create(messages=[], stream=True)
    async with stream:
        async for _chunk in stream:
            break

    [turn] = recorder.turns
    assert not turn.completed
    assert turn.end is not None
    assert turn.arguments_text == '{"code": "a'
    assert turn.usage_output is None
    assert turn.est_output == count_tokens('{"code": "a')
    assert fake.chat.completions.streams[0].closed


async def test_each_create_call_starts_a_new_turn():
    fake = FakeClient([[content_chunk("a")], [content_chunk("b")]])
    recorder = RunRecorder()
    client = InstrumentedClient(fake, recorder)

    for _ in range(2):
        stream = await client.chat.completions.create(messages=[], stream=True)
        async with stream:
            [_ async for _ in stream]

    assert [t.index for t in recorder.turns] == [0, 1]
    assert [t.content_text for t in recorder.turns] == ["a", "b"]


def test_only_long_tool_messages_are_truncated():
    messages = [
        {"role": "user", "content": "u" * 50},
        {"role": "tool", "tool_call_id": "a", "content": "t" * 50},
        {"role": "tool", "tool_call_id": "b", "content": "x" * 10},
    ]

    capped, truncated = truncate_tool_results(messages, limit=20)

    assert truncated == 1
    assert capped[0] == messages[0]
    assert capped[1]["content"].startswith("t" * 20)
    assert "30 of 50 characters omitted" in capped[1]["content"]
    assert capped[2] == messages[2]
    assert messages[1]["content"] == "t" * 50  # the agent's list is untouched


async def test_requests_carry_capped_tool_results_and_count_them():
    fake = FakeClient([[content_chunk("ok")]])
    recorder = RunRecorder()
    client = InstrumentedClient(fake, recorder, tool_result_limit=5)
    messages = [{"role": "tool", "tool_call_id": "a", "content": "0123456789"}]

    stream = await client.chat.completions.create(messages=messages, stream=True)
    async with stream:
        [_ async for _ in stream]

    sent = fake.chat.completions.calls[0]["messages"][0]["content"]
    assert sent.startswith("01234\n[truncated: 5 of 10")
    assert recorder.turns[0].truncated_tool_results == 1


async def test_generation_end_and_upstream_provider_are_recorded():
    first = tool_chunk('{"code": "a')
    first.provider = "DeepInfra"
    fake = FakeClient([[first, tool_chunk('b"}'), FakeChunk(usage=FakeUsage(1, 2))]])
    recorder = RunRecorder()
    client = InstrumentedClient(fake, recorder)

    stream = await client.chat.completions.create(messages=[], stream=True)
    async with stream:
        [_ async for _ in stream]

    [turn] = recorder.turns
    assert turn.upstream_provider == "DeepInfra"
    assert turn.first_arguments <= turn.last_arguments == turn.last_token
    # The usage chunk carries no text, so generation ended before the stream did.
    assert turn.last_token <= turn.end


async def test_chunk_times_are_arrival_times_not_read_times():
    # IPTC stops reading while a closed `code` block executes; the chunks that
    # arrived meanwhile must not look as if the model generated them late.
    fake = FakeClient([[tool_chunk('{"code": "x"'), tool_chunk("}")]])
    recorder = RunRecorder()
    client = InstrumentedClient(fake, recorder)

    stream = await client.chat.completions.create(messages=[], stream=True)
    async with stream:
        async for _chunk in stream:
            await asyncio.sleep(0.2)

    [turn] = recorder.turns
    assert turn.arguments_text == '{"code": "x"}'
    assert turn.last_token == turn.last_arguments
    assert turn.last_token < 0.1
    assert turn.end >= 0.4


async def test_a_failing_stream_raises_to_the_reader():
    class Failing(FakeStream):
        async def __aiter__(self):
            yield content_chunk("a")
            raise RuntimeError("stream broke")

    class FailingCompletions(FakeCompletions):
        async def create(self, **kwargs):
            return Failing([])

    fake = FakeClient([])
    fake.chat.completions = FailingCompletions([])
    recorder = RunRecorder()
    client = InstrumentedClient(fake, recorder)

    stream = await client.chat.completions.create(messages=[], stream=True)
    with pytest.raises(RuntimeError, match="stream broke"):
        async with stream:
            [_ async for _ in stream]

    [turn] = recorder.turns
    assert turn.content_text == "a"
    assert not turn.completed
