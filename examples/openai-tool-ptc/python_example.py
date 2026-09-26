# Programmatic Tool Calling: jitgen executes the Python `code` argument of a
# streamed `eval` tool call statement by statement, while it is still arriving.
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from jitgen import InProcPythonExecutor, Session, create_python_session
from utils import eval_tool, lookup_price, run

SYSTEM_PROMPT = (
    "You solve tasks by writing Python code and calling the `eval` tool to "
    "run it — never compute or guess a numeric result yourself. Inside "
    "that code, an `async def lookup_price(symbol: str) -> float` "
    "function is already available (a Programmatic Tool Calling helper — "
    "do not define or import it, just call it) that returns the current "
    "price of a ticker symbol. It is a coroutine function, so call it as "
    "`await lookup_price(symbol)` — top-level `await` is allowed directly "
    "in this code, do not wrap it in `async def main(): ...` or "
    "`asyncio.run(...)`. Write out a separate `await lookup_price(...)` "
    "call followed by its own `print()` for each symbol individually, one "
    "after another — do NOT use a `for`/`while` loop, a comprehension, or "
    "`asyncio.gather` to iterate over the symbols, even though a loop "
    "would be shorter. Write the code as a flat sequence of top-level "
    "statements (no function/class/loop definitions of your own), and "
    "print every value you need with print() as soon as it is computed."
)

USER_TASK = (
    "Look up the current price of AAPL, MSFT, GOOG, AMZN, and NVDA with "
    "`lookup_price`, then compute and print the total cost of buying 10 "
    "shares of each."
)

EVAL_TOOL = eval_tool(
    description=(
        "Execute Python code and return what it prints. Use this for any "
        "computation instead of answering from memory."
    ),
    code_description=(
        "Python source code to execute. An "
        "`async def lookup_price(symbol: str) -> float` function is already "
        "available in this environment — call it with `await "
        "lookup_price(...)`, top-level `await` works directly here. It needs "
        "no import or definition."
    ),
)


def make_executor() -> InProcPythonExecutor:
    """Build a Python executor whose code can call `lookup_price`."""
    return InProcPythonExecutor(tools={"lookup_price": lookup_price})


@asynccontextmanager
async def open_session() -> AsyncGenerator[Session]:
    """Open a Python session backed by a fresh `make_executor()`.

    Yields:
        The session; its executor is closed on exit.
    """
    executor = make_executor()
    async with executor, create_python_session(executor=executor) as session:
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
