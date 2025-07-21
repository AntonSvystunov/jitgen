from .parser import JITGenParser
from jitgen.prebuilt.python import create_python_jitgen


def create_python_jitgen_parser() -> JITGenParser:
    return JITGenParser(
        jit_gen=create_python_jitgen(),
        start_marker="```python",
        end_marker="```",
    )
