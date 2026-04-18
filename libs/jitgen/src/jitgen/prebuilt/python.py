from lark import Lark
from lark.indenter import PythonIndenter

from jitgen_core import BaseExecutor, Session

from jitgen.executors.python import InProcPythonExecutor
from jitgen.extractors.python import PythonLarkExtractor

_PYTHON_PARSER: Lark = Lark.open_from_package(
    "lark",
    "python.lark",
    ["grammars"],
    parser="lalr",
    postlex=PythonIndenter(),
    start="file_input",
    propagate_positions=True,
)


def create_python_jitgen(executor: BaseExecutor | None = None) -> Session:
    """Build a :class:`~jitgen_core.Session` wired for Python grammar.

    The *executor* is fully client-owned.  Pass any :class:`~jitgen_core.BaseExecutor`
    implementation (e.g. an OpenSandbox-backed executor).  When *None*, a
    default :class:`~jitgen.executors.python.InProcPythonExecutor` is used.

    Marker stripping is **not** handled here — create a
    :class:`~jitgen.markers.MarkerStripper` client-side and call
    :meth:`~jitgen.markers.MarkerStripper.process` before :meth:`~jitgen_core.Session.push`.

    Example::

        from jitgen.prebuilt.python import create_python_jitgen
        from jitgen.markers import MarkerStripper

        executor = InProcPythonExecutor(timeout=30.0, tools={"open": open})
        session  = create_python_jitgen(executor=executor)
        stripper = MarkerStripper(start='{"code":"', end='"}')
    """
    return Session(
        extractor=PythonLarkExtractor(_PYTHON_PARSER),
        executor=executor if executor is not None else InProcPythonExecutor(),
    )
