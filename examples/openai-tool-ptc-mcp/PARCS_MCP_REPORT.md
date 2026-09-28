# PARCS MCP server: stability report

**Server:** the cluster's public MCP endpoint (legacy HTTP+SSE transport, `http://<host>:8080/sse`) and its web UI
**Observed:** 2026-09-27, over about a dozen agent runs against the live server
**Client:** the MCP Python SDK's `sse_client` + `ClientSession`, one long-lived session per agent run. Tools were called from agent-generated Python (Programmatic Tool Calling) with a local `qwen/qwen3.6-27b` model.

This report lists what made agent runs against PARCS fail or waste time, ordered by impact. Each item gives the evidence, the likely cause where we could only infer it, and a suggested fix. Times are UTC.

## Summary

| # | Issue | Impact | Priority |
|---|---|---|---|
| 1 | Long `run_layer` calls lose the SSE connection after about 300s | Any layer longer than ~5 min can never return; the client loses the result | Critical |
| 2 | After a dropped `run_layer`, the client has no `layerId` to recover with | `get_layer_result` can't help; the only option is re-running the layer | Critical |
| 3 | An aggregation layer hung repeatedly on the cluster | A 3-second job never completed across 5 attempts | Critical |
| 4 | No authentication on a public endpoint that compiles and runs arbitrary C# | Anyone who finds the URL can run code on the cluster | Critical |
| 5 | `run_layer`'s documented reply doesn't match what it returns (field name and casing) | Agents repeatedly failed parsing results, wasting whole turns and cluster runs | High |
| 6 | Failures are returned as successful tool results (`isError` not set) | Clients can't tell a compile error from success without parsing | High |
| 7 | Optional parameters are marked required in the tool schema | Clients are told to pass six arguments where two suffice | Medium |
| 8 | Tool results are JSON inside text, with no `outputSchema` | Clients must guess types at each nesting level | Medium |
| 9 | Legacy HTTP+SSE transport | Deprecated in the MCP spec; no resumability for long calls | Medium |
| 10 | Stale documentation (CPU size, cold start time) | Clients size timeouts and jobs wrongly | Low |

## 1. Long `run_layer` calls lose the SSE connection after about 300s (critical)

**Evidence.** In one run, five consecutive `run_layer` calls for the same aggregation layer each ended with the SSE connection closing, always 290-300s after the call started:

| Call started | Connection lost | Duration |
|---|---|---|
| 14:23:33 | 14:28:33 | 300s |
| 14:28:42 | 14:33:36 | 294s |
| 14:33:41 | 14:38:38 | 297s |
| 14:39:14 | 14:44:04 | 290s |

A fifth attempt, started at 14:44:14, was cut off when the run was stopped. Earlier runs also saw `Connection closed` and `httpx` `ReadTimeout` errors on this connection.

**Likely cause.** `run_layer` blocks until the layer completes and sends nothing on the SSE stream meanwhile. The near-constant duration suggests a 5-minute idle timeout somewhere on the path: Kestrel, an ingress or load balancer, or the SSE stream itself.

**Why this is critical.** Cold starts are already close to that limit. First layers took 160-287s while KEDA started 16-21 worker pods (287s on 2026-09-27 at 12:13). A first layer that needs a little more scale-up can never return a result over this transport, however healthy the cluster is.

**Fix (any of these, ideally the first two together):**
- Send periodic MCP progress notifications (`notifications/progress`, e.g. every 15-30s with workers finished / total) while `run_layer` waits. This keeps the stream active and gives clients real progress.
- Send SSE keep-alive comments (`: keepalive\n\n`) at an interval well below every proxy's idle timeout.
- Raise or remove idle timeouts on every hop (Kestrel `KeepAliveTimeout`/request timeouts, ingress `proxy-read-timeout`, cloud load balancer backend timeout) to cover the longest expected layer.

## 2. A dropped `run_layer` leaves the client without a `layerId` (critical)

**Evidence.** The `run_layer` description tells clients that if the connection drops, the reply has status `Running` and they should poll `get_layer_result(layerId)`. But when the connection drops mid-call, the client never receives that reply, so it has no `layerId`. In the run above, the agent could only re-issue `run_layer`. `list_sessions` lists sessions, not layers, so it doesn't help either.

**Fix:**
- Expose a non-blocking pair: `submit_layer(...) -> { layerId }` returning immediately, plus the existing `get_layer_result(layerId)` for polling. The `parcs8` repo's own agent tooling (`misc/mcp_eval/mcp_eval/tools.py`) still lists `submit_layer`, but the live server doesn't offer it.
- Make `run_layer` idempotent: accept an optional client-supplied `requestId`, and have a repeated call with the same id return the existing layer's status or result instead of starting a new layer.
- Add `list_layers(sessionId)` so a client can find layers it started after a disconnect.

## 3. An aggregation layer hung on the cluster (critical)

**Evidence.** In session `6fb8068b07a04ea5aa03fcc2e21fe6e4`:
- The first layer `86ffbf53f401489cbfcfca49888f06b0` completed in 3.1s (16 workers).
- Every `run_layer(parallelism=1, previousLayerId="86ffbf53f401489cbfcfca49888f06b0")` then hung until the connection dropped (item 1), five times between 14:23 and 14:49.
- Meanwhile `get_layer_result("86ffbf53…")` returned `Completed` instantly, `get_cluster_info` responded normally, and a similar aggregation layer in another session had completed in 21s minutes earlier. Aggregation layers normally finish in 0.3-5s.

**We can't see the cause from the client side.** It's worth checking server and pod logs for these calls: was a worker pod ever scheduled, did the previous-layer lookup for `86ffbf53…` stall, did the Pub/Sub message get lost?

**Fix:**
- Enforce a server-side maximum layer duration and per-worker timeout, and fail the layer with a clear `errorMessage` (e.g. "worker 0 not scheduled after 120s") instead of waiting forever.
- Show stuck or pending workers in `get_layer_result` and the web UI (scheduled / running / finished per worker).

## 4. No authentication on a public code-execution endpoint (critical, security)

**Evidence.** We connected to the cluster's MCP endpoint from the public internet with no credentials, and compiled and ran arbitrary C# on the cluster. The traffic is also unencrypted HTTP.

**Fix:**
- Require authentication: at minimum a bearer token checked on every request, preferably the OAuth flow the MCP spec defines for HTTP transports.
- Serve over TLS.
- Restrict network access (allow-listed IPs or a private endpoint) until auth is in place.
- Apply per-client quotas (sessions, layers, parallelism, CPU time) so one client can't monopolize or scale the cluster.
- Sandbox the compiled code: no network egress beyond what jobs need, no access to cluster credentials, service accounts or the Kubernetes API.

## 5. `run_layer`'s documented reply doesn't match the actual reply (high)

**Evidence.** The tool description says:

```
Returns on success:
  { layerId, status:'Completed', totalElapsedSeconds, successCount, failureCount,
    results: [ { workerIndex, success, outputData, errorMessage, elapsedSeconds } ] }
```

The actual reply nests the workers under a singular `result` key, in PascalCase:

```json
{"layerId": "...", "sessionId": "...", "status": "Completed", "submittedAt": "...", "completedAt": "...",
 "totalElapsedSeconds": 31.5, "successCount": 16, "failureCount": 0,
 "result": {"SessionId": "...", "LayerId": "...", "TotalElapsedSeconds": 31.5, "AnyFailures": false,
            "Results": [{"WorkerIndex": 0, "Success": true, "OutputData": "{\"count\":30005}",
                         "ErrorMessage": null, "Metadata": {}, "ElapsedSeconds": 1.2}]}}
```

The same PascalCase envelope is what the next layer receives in `input.PreviousLayerResultJson`, and what `get_layer_result` returns under `result`.

**Impact.** This was the most common cause of wasted work in our runs. Agents following the description wrote `reply["results"]` (a Python `KeyError`), or `GetProperty("results")` / `GetProperty("success")` in C#, which throws `KeyNotFoundException` ("The given key was not present in the dictionary") in the aggregation layer. Each mistake cost a model turn, and usually a full re-run of the job: one run called `create_session` 8 times and re-ran the first layer 7 times.

**Fix:**
- Serialize every payload with one naming policy. camelCase (`JsonSerializerDefaults.Web`) would match the rest of the reply. That covers the stored layer output, `PreviousLayerResultJson`, and the `result` field.
- Either flatten the reply to match the documented shape (`results` at the top level), or update the description to show the real shape, including that `OutputData` is a JSON string the worker returned.
- Add a C# example to the `create_session` description showing how an aggregation layer reads `PreviousLayerResultJson`.

## 6. Failures are returned as successful tool results (high)

**Evidence.** A compile failure returns a normal result with text `{"error":"Compilation failed:\n  [CS0426] ..."}`. Calling `run_layer` with an unknown session returns `{"error":"Session '…' not found."}`. In both cases the MCP result's `isError` flag is not set.

**Impact.** Generic MCP clients, and agents, treat these as successes. In one run the agent indexed `["sessionId"]` on a compile-error reply, crashed with `KeyError`, and never saw the compiler diagnostics.

**Fix.** Set `isError: true` on tool results for compile failures, unknown sessions or layers, invalid parameters, and failed layers. Keep the diagnostic text in the content. This is how the MCP spec distinguishes tool-level failures from successes.

## 7. Optional parameters are marked required (medium)

**Evidence.** `run_layer`'s input schema lists all six parameters as `required`, while four of them (`previousLayerId`, `customData`, `parameters`, `datasetUrl`) document `(Default value: null)`. The server does accept calls that omit them: we tested with only `sessionId` and `parallelism`. But schema-driven clients show all six as mandatory.

**Fix.** Remove optional parameters from `required`. With the C# MCP SDK this usually means declaring them with default values in the tool method (for example `string? previousLayerId = null`) on a version that honors defaults, or overriding the generated schema.

## 8. JSON in text, no `outputSchema` (medium)

**Evidence.** Every tool returns one `TextContent` holding JSON, with no `structuredContent` and no `outputSchema`. Clients must work out on their own that `result` is an object but `OutputData` is a string holding JSON. Our agents got this wrong in both directions: indexing an unparsed string, and calling `json.loads` on an already-parsed object.

**Fix.** Declare an `outputSchema` for each tool and return `structuredContent` alongside the text (MCP 2025-06-18). Consider returning `OutputData` as a JSON object rather than a string when the worker produced valid JSON.

## 9. Legacy HTTP+SSE transport (medium)

The HTTP+SSE transport was deprecated in MCP spec 2025-03-26 in favor of Streamable HTTP. Streamable HTTP supports resuming a stream after a disconnect (`Last-Event-ID`), which addresses item 1 at the protocol level, and it's what current clients expect. Offering a Streamable HTTP endpoint (e.g. `/mcp`) alongside `/sse` would let clients migrate.

## 10. Stale documentation (low)

- The skill documentation says each daemon gets 500m CPU; `get_cluster_info` reports `daemonCpuRequestMillicores: 250`.
- It gives 60-90s for the first layer's cold start; we measured 160-287s with 16-21 workers. Clients size their timeouts from these numbers.
- `maxParallelism` changes as the cluster autoscales (we saw 21, 28 and 35 on the same day). Documenting that it reflects current, not maximum, capacity would help clients size jobs.

## Things that worked well

- `create_session` compiles quickly (0.2-0.5s) with clear `CSxxxx` diagnostics.
- Warm layers are fast: 3s for 16 sieving workers, 0.3-5s for aggregation.
- `get_cluster_info`, `get_layer_result` and `list_sessions` responded instantly throughout, including right after reconnecting.
- Sessions and layers survived client reconnects: after a dropped connection, `get_layer_result` on an earlier layer still worked on the new connection.

## Reproducing issue 1

We haven't run this exact test; it isolates the pattern above from the cluster's own behavior.

1. Connect with any MCP client over `/sse`.
2. `create_session` with a body whose first layer sleeps: `await Task.Delay(TimeSpan.FromMinutes(6), ct); return AgentLayerResult.Ok("{}");`
3. Call `run_layer(sessionId, parallelism=1)`.
4. Expected: a `Completed` reply after about 6 minutes. If the idle-timeout explanation is right, the connection instead closes after about 300s and the client never learns the `layerId`.
