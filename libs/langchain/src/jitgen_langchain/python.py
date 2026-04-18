from jitgen.executors.python import InProcPythonExecutor
from jitgen.markers import MarkerStripper
from jitgen.prebuilt.python import create_python_jitgen

from .parser import JITGenParser


def create_python_jitgen_parser() -> JITGenParser:
    """Convenience factory: Python JITGen session wired with markdown code-fence markers.

    Uses a default :class:`~jitgen.executors.python.InProcPythonExecutor` and
    strips standard triple-backtick ``python`` / closing triple-backtick fences.

    For custom executors, timeout, tools, or markers, build the components
    directly::

        executor = InProcPythonExecutor(timeout=30.0, tools={"open": open})
        session  = create_python_jitgen(executor=executor)
        stripper = MarkerStripper(start="```python", end="```")
        parser   = JITGenParser(session=session, stripper=stripper)
    """
    session = create_python_jitgen(executor=InProcPythonExecutor())
    stripper = MarkerStripper(start="```python", end="```")
    return JITGenParser(session=session, stripper=stripper)
