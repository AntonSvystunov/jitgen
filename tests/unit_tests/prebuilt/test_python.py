"""Tests for jitgen.prebuilt.python module."""

from jitgen.prebuilt.python import create_python_jitgen
from jitgen.core.jit import JITGen
from jitgen.executors.python import InProcPythonExecutor


def test_create_python_jitgen():
    """Test the create_python_jitgen factory function."""
    jitgen = create_python_jitgen()
    
    assert isinstance(jitgen, JITGen)
    assert jitgen.interpreter_type == InProcPythonExecutor
    assert len(jitgen.indentation_tokens) == 3
    assert "_DEDENT" in jitgen.indentation_tokens
    assert "_NEWLINE" in jitgen.indentation_tokens
    assert "$END" in jitgen.indentation_tokens
    assert jitgen.parser is not None

