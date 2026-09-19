import json

import pytest
from jitgen import JitGenError, StreamDriver, create_python_session

from jitgen_openai import OpenAIToolCallSegmenter


async def test_streams_tool_call_arguments_through_a_real_python_session():
    payload = json.dumps(
        {"language": "python", "code": "x = 1 + 1\nprint(x)\nprint('done')"}
    )
    # Split into small, arbitrarily-sized deltas, as a real streaming API would.
    deltas = [payload[i : i + 3] for i in range(0, len(payload), 3)]

    async with create_python_session() as session:
        driver = StreamDriver(session, OpenAIToolCallSegmenter(property_name="code"))

        stdout = ""
        for delta in deltas:
            for output in await driver.apush(delta):
                stdout += output
        for output in await driver.afinish():
            stdout += output

    assert stdout == "2\ndone\n"
    assert driver.has_error is False


async def test_reports_execution_errors_from_streamed_code():
    payload = json.dumps({"code": "raise ValueError('boom')"})

    async with create_python_session() as session:
        driver = StreamDriver(session, OpenAIToolCallSegmenter(property_name="code"))

        # The whole tool call arrives in one chunk here, so the code's
        # closing quote — and thus its execution — happens inside this one
        # `apush` call rather than being deferred to `afinish`.
        with pytest.raises(JitGenError, match="boom"):
            await driver.apush(payload)
