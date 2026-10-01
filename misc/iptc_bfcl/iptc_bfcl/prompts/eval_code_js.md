How to write the `code`:

- Make every call the request needs in a single `eval` call. Start another `eval` call only to recover from an error; constants from earlier `eval` calls still exist, and redeclaring one is allowed.
- Write one flat sequence of top-level statements, one call per statement: no functions of your own, no loops over calls, no `Promise.all`, no `.then()` chains. End every statement with a semicolon and declare variables with `const`.
- Call each function with `await`, directly at top level (no `async function main() {...}` wrapper), passing its arguments as one object: `await some_function({name: "value", count: 3})`.
- The code runs in a bare JavaScript engine, not Node.js or a browser: there is no `require`, `import`, `process`, `fetch` or `setTimeout`, and `console.log` is the only output.
- After each call, log its result.
- On a retry, don't repeat a call that already succeeded.

Skeleton:

```javascript
const first = await some_function({name: "value", count: 3});
console.log(first);
const second = await other_function({items: ["a", "b"], enabled: true});
console.log(second);
```
