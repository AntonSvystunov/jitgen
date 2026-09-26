# Same as python_example.py, but the `eval` tool's `code` argument is
# JavaScript, executed by jitgen-js's QuickJS-backed session.
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from jitgen import Session
from jitgen_js import QuickJsExecutor, create_javascript_session
from utils import eval_tool, lookup_price, run

SYSTEM_PROMPT = (
    "You solve tasks by writing JavaScript code and calling the `eval` "
    "tool to run it — never compute or guess a numeric result yourself. "
    "Inside that code, an async `lookupPrice(symbol)` function is already "
    "available (a Programmatic Tool Calling helper — do not define or "
    "import it, just call it) that returns the current price of a ticker "
    "symbol. It returns a Promise, so call it as `await "
    "lookupPrice(symbol)` — top-level `await` is allowed directly in this "
    "code, do not wrap it in an `async function` or a `.then()` chain. "
    "Write out a separate `await lookupPrice(...)` call followed by its "
    "own `console.log()` for each symbol individually, one after another "
    "— do NOT use a `for`/`while` loop, `.map()`/`.forEach()`, or "
    "`Promise.all` to iterate over the symbols, even though a loop would "
    "be shorter. Write the code as a flat sequence of top-level "
    "statements (no function/loop declarations of your own — a call to a "
    "function declared later in the same script will fail), and log "
    "every value you need with console.log() as soon as it is computed."
)

USER_TASK = (
    "Look up the current price of AAPL, MSFT, GOOG, AMZN, and NVDA with "
    "`lookupPrice`, then compute and log the total cost of buying 10 "
    "shares of each."
)

EVAL_TOOL = eval_tool(
    description=(
        "Execute JavaScript code and return what it logs via console.log(). "
        "Use this for any computation instead of answering from memory."
    ),
    code_description=(
        "JavaScript source code to execute. An async `lookupPrice(symbol)` "
        "function is already available in this environment — call it with "
        "`await lookupPrice(...)`, top-level `await` works directly here. It "
        "needs no import or definition."
    ),
)


def make_executor() -> QuickJsExecutor:
    """Build a JavaScript executor whose code can call `lookupPrice`."""
    return QuickJsExecutor(tools={"lookupPrice": lookup_price})


@asynccontextmanager
async def open_session() -> AsyncGenerator[Session]:
    """Open a JavaScript session backed by a fresh `make_executor()`.

    Yields:
        The session; its executor is closed on exit.
    """
    executor = make_executor()
    async with executor, create_javascript_session(executor=executor) as session:
        yield session


async def main() -> None:
    load_dotenv()
    await run(
        USER_TASK,
        system_prompt=SYSTEM_PROMPT,
        tool=EVAL_TOOL,
        open_session=open_session,
    )


if __name__ == "__main__":
    asyncio.run(main())
