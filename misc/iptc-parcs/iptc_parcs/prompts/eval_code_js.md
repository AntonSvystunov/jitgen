How to write the `code`:

- Put the whole job (cluster info, C# source, session, every layer, final log) into a single `eval` call. Start another `eval` call only to recover from an error; constants from earlier `eval` calls still exist, and redeclaring one is allowed.
- Write one flat sequence of top-level statements: no functions of your own, no loops over tool calls, no `Promise.all`, no `.then()` chains. End every statement with a semicolon and declare variables with `const`.
- Call each helper with `await`, directly at top level (no `async function main() {...}` wrapper), passing its arguments as one object: `await run_layer({sessionId: sessionId, parallelism: info.workerNodeCount})`.
- Every helper returns a string of JSON: parse it exactly once, in the same statement as the call, with `JSON.parse`, and keep the resulting object in a named constant. `layer.result` is then already an object; each `OutputData` is still a string and needs `JSON.parse`.
- Put the C# code in a raw template literal, `` const source = String.raw`...`; ``, so its backslashes stay as written. The C# must not contain a backtick or the sequence `${`.
- The code runs in a bare JavaScript engine, not Node.js or a browser: there is no `require`, `import`, `process`, `fetch` or `setTimeout`, and `console.log` is the only output. Log objects with `JSON.stringify(...)`; a bare object prints as `[object Object]`.
- After each call, log a short summary (status, ids, `totalElapsedSeconds`, `successCount`/`failureCount`, or the error), never a whole result.
- On a retry, don't repeat a call that already succeeded: continue from the constants you have.

Skeleton:

```javascript
const info = JSON.parse(await get_cluster_info());
console.log(JSON.stringify(info));

const source = String.raw`
...C# body...
`;
const session = JSON.parse(await create_session({sourceCode: source}));
console.log(session.error || session.sessionId);
const sessionId = session.sessionId;

const layer1 = JSON.parse(
  await run_layer({sessionId: sessionId, parallelism: info.workerNodeCount, parameters: {start: "1000", end: "2000"}})
);
console.log(layer1.status, layer1.totalElapsedSeconds, layer1.successCount, layer1.failureCount);
const layer1Id = layer1.layerId;

const layer2 = JSON.parse(
  await run_layer({sessionId: sessionId, parallelism: 1, previousLayerId: layer1Id})
);
console.log(layer2.status, layer2.totalElapsedSeconds);
const answer = JSON.parse(layer2.result.Results[0].OutputData);
console.log(JSON.stringify(answer));
```
