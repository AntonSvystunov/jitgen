from functools import cache

from lark import Lark
from lark.indenter import PythonIndenter

from jitgen.base import BaseExecutor
from jitgen.executors.python import InProcPythonExecutor
from jitgen.extractors.python import PythonLarkExtractor
from jitgen.session import Session


@cache
def python_parser() -> Lark:
    """Build the shared Python LALR parser once, on first use.

    Deliberately lazy: this factory is re-exported from `jitgen`, and building
    the grammar eagerly would charge every `import jitgen` for it — including
    imports that never touch Python extraction. `Lark.parse` is stateless, so
    one instance serves every session.

    Returns:
        A `Lark` parser configured for `PythonLarkExtractor`.
    """
    return Lark.open_from_package(
        "lark",
        "python.lark",
        ["grammars"],
        parser="lalr",
        postlex=PythonIndenter(),
        start="file_input",
        propagate_positions=True,
    )


def create_python_session(executor: BaseExecutor | None = None) -> Session:
    """Build a `Session` (`jitgen.session.Session`) wired for the Python grammar.

    Code extraction is not handled here — pair the session with a
    `CodeSegmenter` (`jitgen.base.CodeSegmenter`) via `StreamDriver`
    (`jitgen.driver.StreamDriver`):

    ```python
    async with create_python_session() as session:
        driver = StreamDriver(session, markdown_code("python"))
    ```

    Args:
        executor: Where code runs — a sandbox, a subprocess. When `None`, an
            `InProcPythonExecutor` (`jitgen.executors.python.InProcPythonExecutor`)
            is created *and owned* by the session, so `await session.aclose()`
            shuts down its worker thread; an executor you pass in stays yours
            to close.

    Returns:
        A `Session` ready to receive pushed source code.
    """
    owns_executor = executor is None
    return Session(
        extractor=PythonLarkExtractor(python_parser()),
        executor=executor if executor is not None else InProcPythonExecutor(),
        owns_executor=owns_executor,
    )
