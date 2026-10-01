How to write the `code`:

- Put the whole job (cluster info, C# source, session, every layer, final print) into a single `eval` call. Start another `eval` call only to recover from an error; variables from earlier `eval` calls still exist.
- Write one flat sequence of top-level statements: no functions of your own, no loops over tool calls, no `asyncio.gather`.
- Every helper returns a `str` of JSON: parse it exactly once, in the same statement as the call, and keep the resulting `dict` in a named variable. `layer["result"]` is then already a `dict`; each `OutputData` is still a `str` and needs `json.loads`.
- Put the C# code in a raw string, `source = r"""..."""`, not an f-string.
- After each call, print a short summary (status, ids, `totalElapsedSeconds`, `successCount`/`failureCount`, or the error), never a whole result.
- On a retry, don't repeat a call that already succeeded: continue from the variables you have.

Skeleton:

```python
import json

info = json.loads(await get_cluster_info())
print(info)

source = r"""
...C# body...
"""
session = json.loads(await create_session(sourceCode=source))
print(session.get("error") or session["sessionId"])
session_id = session["sessionId"]

layer1 = json.loads(
    await run_layer(
        sessionId=session_id,
        parallelism=info["workerNodeCount"],
        parameters={"start": "1000", "end": "2000"},
    )
)
print(
    layer1["status"],
    layer1["totalElapsedSeconds"],
    layer1["successCount"],
    layer1["failureCount"],
)
layer1_id = layer1["layerId"]

layer2 = json.loads(
    await run_layer(sessionId=session_id, parallelism=1, previousLayerId=layer1_id)
)
print(layer2["status"], layer2["totalElapsedSeconds"])
answer = json.loads(layer2["result"]["Results"][0]["OutputData"])
print(answer)
```
