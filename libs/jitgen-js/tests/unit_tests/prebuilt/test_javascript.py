import threading

from jitgen import Session

from jitgen_js.executors.quickjs import QuickJsExecutor
from jitgen_js.prebuilt.javascript import create_javascript_session


def test_create_javascript_session_returns_session():
    session = create_javascript_session()
    assert isinstance(session, Session)


def test_default_executor_is_owned_by_the_session():
    """A session that built its own executor must also shut it down."""
    session = create_javascript_session()
    assert isinstance(session.executor, QuickJsExecutor)
    assert session._owns_executor is True


def test_injected_executor_stays_the_callers():
    custom = QuickJsExecutor(timeout=5.0)
    session = create_javascript_session(executor=custom)
    assert session.executor is custom
    assert session._owns_executor is False


async def test_executes_simple_statement():
    async with create_javascript_session() as session:
        session.push("console.log('hello');\n")
        assert await session.result() == "hello\n"


async def test_maintains_repl_state():
    async with create_javascript_session() as session:
        session.push("var x = 42;\n")
        session.push("console.log(x);\n")
        assert await session.result() == "42\n"


async def test_with_injected_tool():
    executor = QuickJsExecutor(tools={"add": lambda a, b: a + b})
    async with executor, create_javascript_session(executor=executor) as session:
        session.push("console.log(add(2, 3));\n")
        assert await session.result() == "5\n"


async def test_closing_the_session_stops_the_owned_worker_thread():
    """The leak this fixes: every executor starts a worker thread nobody was closing.

    Counted by name rather than with `active_count()` — `aclose` joins via
    `asyncio.to_thread`, whose pool worker outlives the call.
    """

    def executor_threads() -> int:
        return sum(t.name == "jitgen-quickjs-executor" for t in threading.enumerate())

    before = executor_threads()
    async with create_javascript_session() as session:
        session.push("console.log(1);\n")
        await session.result()
        assert executor_threads() == before + 1
    assert executor_threads() == before
