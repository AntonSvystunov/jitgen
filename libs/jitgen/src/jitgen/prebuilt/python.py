from lark import Lark
from lark.indenter import PythonIndenter
from jitgen_core import JITGen
from jitgen_core.v2 import JITGenV2

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