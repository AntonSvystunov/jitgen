# IPTC agent over the read-only `mcp-server-fetch` MCP server: the model's
# `eval` code fetches pages while the rest of the code is still streaming.
# `--language javascript` has the model write JavaScript, run in QuickJS.
import argparse
import asyncio

from jitgen_openai import CodeLanguage, McpToolBridge
from mcp import StdioServerParameters

from utils import add_language_argument, run_agent

# Both prompts insist on one `eval` call: models otherwise tend to fetch one
# URL per call, or to compare the results in a second call, and every extra
# call costs a full model round trip.
PYTHON_SYSTEM_PROMPT = (
    "You solve tasks by writing Python code and calling the `eval` tool to "
    "run it — never answer from memory. Make exactly ONE `eval` call that "
    "does the whole task: every tool call, every computation, and the final "
    "statements that print the result the task asks for (for example, which "
    "page is longer and by how much). Do not split the work across calls — "
    "not one call per URL, and not a second call to inspect or compare "
    "earlier results; variables are already in scope, so compute on them in "
    "the same code. Call `eval` again only if the first call failed, and "
    "then only for the part that failed. Any async helper the tool's "
    "description lists is already available in this environment; call it "
    "with `await`, directly at top level (no `async def main(): ...`/"
    "`asyncio.run(...)` wrapper) — do not define or import it yourself. "
    "Write out a separate `await fetch(...)` call followed by its own "
    "`print()` for each URL individually, one after another — do NOT use a "
    "`for`/`while` loop, a comprehension, or `asyncio.gather` to iterate "
    "over the URLs, even though a loop would be shorter. `fetch` returns at "
    "most `max_length` characters (5000 by default), so pass "
    "`max_length=200000` to get whole pages. Print short summaries such as "
    "lengths, never whole pages. Once the `eval` result arrives, reply in "
    "plain text with the answer instead of calling `eval` again."
)

JAVASCRIPT_SYSTEM_PROMPT = (
    "You solve tasks by writing JavaScript code and calling the `eval` tool "
    "to run it — never answer from memory. Make exactly ONE `eval` call that "
    "does the whole task: every tool call, every computation, and the final "
    "statements that log the result the task asks for (for example, which "
    "page is longer and by how much). Do not split the work across calls — "
    "not one call per URL, and not a second call to inspect or compare "
    "earlier results; variables are already in scope, so compute on them in "
    "the same code. Call `eval` again only if the first call failed, and "
    "then only for the part that failed. Any async helper the tool's "
    "description lists is already available in this environment; call it "
    "with `await`, directly at top level (no `async function main() {...}` "
    "wrapper, no `.then()` chains) — do not define or import it yourself. "
    "Pass a helper's arguments as one object, e.g. "
    '`await fetch({url: "...", max_length: 200000})`; every helper returns '
    "a string. The code runs in a bare JavaScript engine, not Node.js or a "
    "browser: there is no `require`, `import`, `process` or `setTimeout`, "
    "and `console.log` is the only output. Declare variables with `const`, "
    "end every statement with a semicolon, and define no functions of your "
    "own. Write out a separate `const page = await fetch({...});` statement "
    "followed by its own `console.log(...)` for each URL individually, one "
    "after another — do NOT use a `for`/`while` loop, `.map`/`.forEach`, or "
    "`Promise.all` to iterate over the URLs, even though a loop would be "
    "shorter. `fetch` returns at most `max_length` characters (5000 by "
    "default), so pass `max_length: 200000` to get whole pages. Log short "
    "summaries such as lengths, never whole pages. Once the `eval` result "
    "arrives, reply in plain text with the answer instead of calling `eval` "
    "again."
)

SYSTEM_PROMPTS: dict[CodeLanguage, str] = {
    "python": PYTHON_SYSTEM_PROMPT,
    "javascript": JAVASCRIPT_SYSTEM_PROMPT,
}

USER_TASK = (
    "Fetch the full content of these two Wikipedia articles: "
    "https://en.wikipedia.org/wiki/Python_(programming_language) and "
    "https://en.wikipedia.org/wiki/JavaScript. Tell me which one's "
    "fetched content is longer, and by how many characters."
)

FETCH_SERVER = StdioServerParameters(command="uvx", args=["mcp-server-fetch"])


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fetch agent.")
    add_language_argument(parser)
    args = parser.parse_args()
    await run_agent(
        McpToolBridge.stdio(FETCH_SERVER),
        USER_TASK,
        system_prompt=SYSTEM_PROMPTS[args.language],
        language=args.language,
    )


if __name__ == "__main__":
    asyncio.run(main())
