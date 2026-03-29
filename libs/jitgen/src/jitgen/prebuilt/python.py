from lark import Lark
from lark.indenter import PythonIndenter
from jitgen_core import JITGen
from jitgen_core.aio import AsyncJITGenSession
from jitgen_core.v2 import JITGenV2
from jitgen_core.v2 import JITGenSession

from jitgen.executors.python import InProcPythonExecutor

kwargs = dict(postlex=PythonIndenter(), start="file_input", propagate_positions=True)
python_parser3 = Lark.open_from_package(
    "lark", "python.lark", ["grammars"], parser="lalr", **kwargs
)

def create_python_jitgen() -> JITGen:
    return JITGen(
        parser=python_parser3,
        interpreter_type=InProcPythonExecutor,
        indentation_tokens={
            "_DEDENT",
            "_NEWLINE",
            "$END",
        },
    )


def create_python_jitgen_v2() -> JITGenV2:
    """Create a push-based JITGenV2 instance configured for Python."""
    return JITGenV2(
        parser=python_parser3,
        interpreter_type=InProcPythonExecutor,
        indentation_tokens={
            "_DEDENT",
            "_NEWLINE",
            "$END",
        },
    )


def create_python_jitgen_session(
    *,
    start_marker: str = "```python",
    end_marker: str = "```",
) -> JITGenSession:
    """Create a stateful marker-aware JITGen session for synchronous workflows."""
    return JITGenSession(
        parser=python_parser3,
        interpreter_type=InProcPythonExecutor,
        indentation_tokens={
            "_DEDENT",
            "_NEWLINE",
            "$END",
        },
        start_marker=start_marker,
        end_marker=end_marker,
    )


def create_python_async_jitgen_session(
    *,
    start_marker: str = "```python",
    end_marker: str = "```",
) -> AsyncJITGenSession:
    """Create a stateful marker-aware JITGen session for async workflows."""
    return AsyncJITGenSession(
        parser=python_parser3,
        interpreter_type=InProcPythonExecutor,
        indentation_tokens={
            "_DEDENT",
            "_NEWLINE",
            "$END",
        },
        start_marker=start_marker,
        end_marker=end_marker,
    )