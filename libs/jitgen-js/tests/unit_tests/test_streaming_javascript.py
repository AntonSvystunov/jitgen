import pytest
from jitgen import StreamDriver
from jitgen.segmenters import markdown_code

from jitgen_js import create_javascript_session

# A realistic script exercising, in one buffer, the exact shapes that broke
# the extractor's original tail-position recovery heuristic when fed
# chunk-by-chunk: a multi-line unclosed function call, a string literal
# containing a brace character, and a try/catch/finally. Chunked feeding
# through the real `StreamDriver` (not direct `extract()` calls with
# hand-picked buffers) is what actually caught those bugs — unit tests
# against the extractor in isolation did not.
_SCRIPT = """Here is some JS:
```javascript
const greeting = "Hello";
console.log(greeting);
if (greeting === "Hello") {
  console.log("matched");
} else {
  console.log("no match");
}
try {
  JSON.parse("{bad json");
} catch (e) {
  console.log("caught: " + e.name);
} finally {
  console.log("cleanup");
}
```
Done.
"""

_EXPECTED_OUTPUT = "Hello\nmatched\ncaught: SyntaxError\ncleanup\n"


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 13, 64])
async def test_streams_a_realistic_script_at_every_chunk_size(chunk_size: int):
    session = create_javascript_session()
    driver = StreamDriver(session, markdown_code("javascript"))

    outputs: list[str] = []
    for i in range(0, len(_SCRIPT), chunk_size):
        outputs.extend(await driver.apush(_SCRIPT[i : i + chunk_size]))
    outputs.extend(await driver.afinish())

    assert "".join(outputs) == _EXPECTED_OUTPUT
    await session.aclose()
