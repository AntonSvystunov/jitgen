import threading

from jitgen import InProcPythonExecutor, Session
from jitgen.prebuilt.python import create_python_session, python_parser


def test_create_python_session_returns_session():
    session = create_python_session()
    assert isinstance(session, Session)


def test_default_executor_is_owned_by_the_session():
    """A session that built its own executor must also shut it down."""
    session = create_python_session()
    assert isinstance(session.executor, InProcPythonExecutor)
    assert session._owns_executor is True


def test_injected_executor_stays_the_callers():
    custom = InProcPythonExecutor(timeout=5.0)
    session = create_python_session(executor=custom)
    assert session.executor is custom
    assert session._owns_executor is False


def test_parser_is_built_once():
    """The grammar build is expensive; `import jitgen` must not pay for it."""
    assert python_parser() is python_parser()
    assert python_parser.cache_info().currsize == 1


async def test_executes_simple_statement():
    async with create_python_session() as session:
        session.push("print('hello')\n")
        assert await session.result() == "hello\n"


async def test_maintains_repl_state():
    async with create_python_session() as session:
        session.push("x = 42\n")
        session.push("print(x)\n")
        assert await session.result() == "42\n"


async def test_with_injected_tool():
    executor = InProcPythonExecutor(tools={"add": lambda a, b: a + b})
    async with executor, create_python_session(executor=executor) as session:
        session.push("print(add(2, 3))\n")
        assert await session.result() == "5\n"


async def test_closing_the_session_stops_the_owned_worker_thread():
    """The leak this fixes: every executor starts a thread nobody was closing.

    Counted by name rather than with `active_count()` — `aclose` joins via
    `asyncio.to_thread`, whose pool worker outlives the call.
    """

    def executor_threads() -> int:
        return sum(t.name == "jitgen-inproc-executor" for t in threading.enumerate())

    before = executor_threads()
    async with create_python_session() as session:
        session.push("print(1)\n")
        await session.result()
        assert executor_threads() == before + 1
    assert executor_threads() == before
