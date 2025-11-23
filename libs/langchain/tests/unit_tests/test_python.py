"""Tests for jitgen.langchain.python module."""

from jitgen_langchain.python import create_python_jitgen_parser
from jitgen_langchain.parser import JITGenParser


def test_create_python_jitgen_parser():
    """Test the create_python_jitgen_parser factory function."""
    parser = create_python_jitgen_parser()
    
    assert isinstance(parser, JITGenParser)
    assert parser.start_marker == "```python"
    assert parser.end_marker == "```"
    assert parser.jit_gen is not None

