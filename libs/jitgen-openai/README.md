# jitgen-openai

An OpenAI tool-call-argument `CodeSegmenter` for [JitGen](../jitgen).

## Why

Most JitGen use cases so far look like a chat completion where the model
writes a fenced ` ```python ` block. The more relevant use case for
grammar-guided *incremental* execution is a model driving a **tool call**
whose argument is source code — an `eval`-style tool, as in LangChain
deepagents' [code-execution
pattern](https://docs.langchain.com/oss/python/deepagents/interpreters).

OpenAI (and OpenAI-compatible) tool-call streaming delivers a function's
`arguments` as successive plain-text deltas of one growing JSON object:

```
{"la
{"language": "python", "co
{"language": "python", "code": "pri
...
```

`OpenAIToolCallSegmenter` scans that raw, growing JSON text and emits the
**decoded** string value of one named property (e.g. `"code"`) as soon as
each character of it is final, so `StreamDriver` can dispatch statements to
the executor while the tool call is still streaming in — before the model
has finished writing the rest of the `arguments` object.

## Usage

```python
from jitgen import StreamDriver, create_python_session
from jitgen_openai import OpenAIToolCallSegmenter

async with create_python_session() as session:
    driver = StreamDriver(session, OpenAIToolCallSegmenter(property_name="code"))

    async for chunk in stream:  # an OpenAI ChatCompletionChunk stream
        tool_calls = chunk.choices[0].delta.tool_calls
        if not tool_calls or tool_calls[0].function is None:
            continue
        arguments_delta = tool_calls[0].function.arguments
        if arguments_delta is None:
            continue

        for output in await driver.apush(arguments_delta):
            print(output, end="")

    for output in await driver.afinish():
        print(output, end="")
```

`feed()` takes plain `str` chunks — the raw `arguments` delta text, same
division of responsibility `MarkerSegmenter` uses for `delta.content`. This
package has no OpenAI SDK dependency; the caller extracts the delta text
from whatever streaming client it uses.

## Contract and limitations

- One instance scans one JSON object's `property_name` at a time. It's safe
  to keep feeding it further JSON afterward — either after `StreamDriver`
  resets it on `end_of_block`, or even concatenated within a single
  `feed()` call (e.g. two tool calls' `arguments` fed back to back) — each
  new `{` restarts the scan.
- `property_name` must name a top-level JSON **string** property. Any other
  properties (of any type — string, number, object, array, bool, null)
  before or after it are correctly skipped, including nested structures and
  strings that happen to contain brace/quote characters or the target
  property's own name as literal text.
- JSON string escapes (`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`,
  `\uXXXX`, including surrogate pairs) are decoded incrementally and can be
  split across any chunk boundary without being misread. A lone,
  never-paired high surrogate is flushed best-effort rather than dropped,
  but this is not a fully RFC-8259-compliant JSON parser — it is scoped to
  what's needed to reliably locate and decode one string field inside a
  growing, not-yet-complete JSON object.
- Malformed input encountered outside the target property (e.g. missing
  colons, unexpected tokens) is handled leniently — the scanner keeps
  advancing rather than raising, since misreading the *framing* around the
  code should never surface as a `SyntaxError` from the code itself.
